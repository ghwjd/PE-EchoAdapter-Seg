"""
Training script of SAMWISE
Modified from DETR (https://github.com/facebookresearch/detr)
"""
# main.py
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from util.misc import on_load_checkpoint
import datasets.samplers as samplers
from datasets import build_dataset
from engine import train_one_epoch
from models.samwise import build_samwise
from os.path import join
import sys
import opts

from torch.utils.tensorboard import SummaryWriter

def main(args):
    utils.init_distributed_mode(args)
    if args.output_dir and utils.get_rank() == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.log_file = join(args.output_dir, 'log.txt')
        with open(args.log_file, 'w') as fp:
            fp.writelines(" ".join(sys.argv)+'\n')
            fp.writelines(str(args.__dict__)+'\n\n')

    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # TensorBoard 로그 디렉토리 지정
    writer = SummaryWriter(log_dir=join(args.output_dir, 'tensorboard')) if utils.is_main_process() else None
    global_step = 0

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_samwise(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        # model_without_ddp = model.module
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu], 
            find_unused_parameters=True
        )
        model_without_ddp = model.module

    n_parameters_tot = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters_tot)

    head = []
    fix = []
    for k, v in model_without_ddp.named_parameters():
        if v.requires_grad:
            head.append(v)
        else:
            fix.append(v)

    print("Trainable parameters: ", sum(p.numel() for p in head))
    print("Parameters fixed: ", sum(p.numel() for p in fix))

    # param_list = [{
    #     'params': head,
    #     'initial_lr': args.lr
    # }]

    # optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)

    # === AdamW 파라미터 그룹: decay vs no_decay 분리 ===
    decay, no_decay = [], []
    for n, p in model_without_ddp.named_parameters():
        if not p.requires_grad:
            continue
        # if n.endswith(".bias") or "norm" in n.lower() or "gating_param" in n or p.ndim == 1:
        lname = n.lower()
        if any(k in lname for k in ['bias', 'norm', 'bn', 'layernorm',
                                'gating_param', 'adapter_tokens']):
            no_decay.append(p)
        else:
            decay.append(p)

    # param_list = [
    #     {'params': decay,    'weight_decay': args.weight_decay, 'initial_lr': args.lr},
    #     {'params': no_decay, 'weight_decay': 0.0,               'initial_lr': args.lr},
    # ]
    optimizer = torch.optim.AdamW(
        [{'params': decay,   'weight_decay': args.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}],
        lr=args.lr
    )

    # optimizer = torch.optim.AdamW(param_list, lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(args.dataset_file, image_set="train", args=args)

    args.batch_size = int(args.batch_size / args.ngpu)
    if args.distributed:
        sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    args.spike_log_path = output_dir / "loss_spike.json"
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint = on_load_checkpoint(model_without_ddp, checkpoint)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if args.resume_optimizer and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print(
                    'Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1


    if args.motion_prompt:
        # initialize projection weights for the MOTION prompt with those of CLS
        mlp_weights = model_without_ddp.sam.sam_prompt_encoder.project_text.state_dict()
        model_without_ddp.sam.sam_prompt_encoder.project_motion_prompts.load_state_dict(mlp_weights)

    print("Start training")
    start_time = time.time()
    log_freq = 50
    if utils.is_main_process():
        (output_dir / "loss_spike.json").write_text("")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
                    model, data_loader_train, optimizer, device, epoch,
                    args.clip_max_norm, lr_scheduler=lr_scheduler, args=args, writer=writer, log_freq=log_freq)
        
        if writer is not None:
            for k, v in train_stats.items():
                writer.add_scalar(f'train/{k}', v, epoch)
            writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)
        global_step += 1

        if args.output_dir:
            print("Save Model")
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters_tot}

        if utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.dataset_file == 'ytvos':
            print("Evaluate on DAVIS: ")
            from inference_davis import eval_davis
            m = model
            out_dir = join(args.output_dir, f'valid_epoch{str(epoch).zfill(2)}')
            eval_davis(args, m, out_dir)
        elif args.dataset_file == 'mevis':
            m = model
            print("Evaluate on MeVis: ")
            from inference_mevis import eval_mevis
            out_dir = join(args.output_dir, f'valid_epoch{str(epoch).zfill(2)}')
            args.split = 'valid_u'
            result = eval_mevis(args, m, out_dir, out_dir)

            if utils.is_main_process():
                out_str = f'Epoch: {epoch}:\nJ: {result[0]},\t F: {result[1]},\t J&F: {result[2]}'
                print(out_str)
                with (output_dir / "log.txt").open("a") as f:
                    f.write(out_str + "\n")
        elif args.dataset_file == 'echonet':
            from inference_echonet import eval_echonet
            out_dir = join(args.output_dir, f'val_epoch{epoch:02d}')
            vis_prefix = join(out_dir, "vis")  # 저장 폴더
            os.makedirs(vis_prefix, exist_ok=True)

            if utils.is_main_process():
                print("Evaluate on EchoNet (val split):")
                # DDP면 model_without_ddp가 더 안전
                result = eval_echonet(args, model_without_ddp, out_dir, vis_prefix)

                out_str = f'Epoch {epoch}: J={result[0]:.3f}, F={result[1]:.3f}, J&F={result[2]:.3f}'
                print(out_str)
                with (output_dir / "log.txt").open("a") as f:
                    f.write(out_str + "\n")
                writer.add_scalar('val/J',  result[0], epoch)
                writer.add_scalar('val/F',  result[1], epoch)
                writer.add_scalar('val/JF', result[2], epoch)

            if args.distributed:
                torch.distributed.barrier()

        elif args.dataset_file == 'camus':   # ★ CAMUS 추가 ★
            print("Evaluate on CAMUS (val split):")
            from inference_camus import eval_camus   # 새 평가 스크립트 필요
            out_dir = join(args.output_dir, f'val_epoch{epoch:02d}')
            result = eval_camus(args, model_without_ddp, out_dir, split='val')

            if utils.is_main_process():
                out_str = f'Epoch {epoch}: J={result[0]:.3f}, F={result[1]:.3f}, J&F={result[2]:.3f}'
                print(out_str)
                with (output_dir / "log.txt").open("a") as f:
                    f.write(out_str + "\n")
                writer.add_scalar('val/J', result[0], epoch)
                writer.add_scalar('val/F', result[1], epoch)
                writer.add_scalar('val/JF', result[2], epoch)
                
            if args.distributed:
                torch.distributed.barrier()

        if utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if writer is not None:   # ← 추가
        writer.close()
    
    os.makedirs("pretrain", exist_ok=True)

    if utils.is_main_process():
        ckpt_name = f"final_model_{args.name_exp}.pth"
        final_path = os.path.join("pretrain", ckpt_name)
        torch.save({'model': model_without_ddp.state_dict()}, final_path)
        # torch.save({'model': model.state_dict()}, final_path)
        tmp_path = final_path + ".tmp"
        torch.save({'model': model_without_ddp.state_dict()}, tmp_path)
        # torch.save({'model': model.state_dict()}, tmp_path)
        os.replace(tmp_path, final_path)
        print(f"[✓] 최종 가중치 저장 완료 ➜ {final_path}")


if __name__ == '__main__':
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser('SAMWISE training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    name_exp = args.name_exp
    args.output_dir = os.path.join(args.output_dir, name_exp)

    main(args)
    