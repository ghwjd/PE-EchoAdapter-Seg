'''
Inference & Evaluation code for SAMWISE on EchoNet dataset
Adapted from inference_mevis.py (SAMWISE)
'''
import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
import util.misc as utils
import os
from PIL import Image
import torch.nn.functional as F
import json
from tqdm import tqdm
from pycocotools import mask as coco_mask
import opts
from models.samwise import build_samwise
from util.misc import on_load_checkpoint
from tools.metrics import db_eval_boundary, db_eval_iou
from datasets.echonet import EchoNetDataset
from datasets.transform_utils import make_coco_transforms
from torch.utils.data import DataLoader
from datasets.transform_utils import vis_add_mask
from tools.colormap import colormap

# Color map for visualization
color_list = colormap().astype('uint8').tolist()

def main(args):
    args.batch_size = 1
    print("Inference only supports batch size = 1") 
    print(args)
    
    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    utils.init_distributed_mode(args)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Save path setup
    output_dir = args.output_dir
    save_path_prefix = os.path.join(output_dir, 'Annotations')
    os.makedirs(save_path_prefix, exist_ok=True)
    args.log_file = os.path.join(args.output_dir, 'log.txt')
    with open(args.log_file, 'w') as fp:
        fp.writelines(" ".join(os.sys.argv)+'\n')
        fp.writelines(str(args.__dict__)+'\n\n')

    save_visualize_path_prefix = os.path.join(output_dir, args.split + '_images')
    if args.visualize:
        os.makedirs(save_visualize_path_prefix, exist_ok=True)

    # Load model
    model = build_samwise(args)
    device = torch.device(args.device)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print(f'Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # Load checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if list(checkpoint['model'].keys())[0].startswith('module'):
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}        
        checkpoint = on_load_checkpoint(model_without_ddp, checkpoint)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print(f'Missing Keys: {missing_keys}')
        print(f'Unexpected Keys: {unexpected_keys}')

    # Run evaluation
    print('Start inference & evaluation on EchoNet')
    J_score, F_score, JF_score = eval_echonet(args, model, save_path_prefix, save_visualize_path_prefix, split=args.split)
    print(f'Final Evaluation Results → J: {J_score:.4f}, F: {F_score:.4f}, J&F: {JF_score:.4f}')


def eval_echonet(args, model, save_path_prefix, save_visualize_path_prefix, split='test'):
    # Load EchoNet dataset
    root = Path(args.echonet_path)
    ds = EchoNetDataset(
        root / split,
        transforms=make_coco_transforms(split, args.max_size, False),
        num_frames=args.num_frames
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda x: x[0], num_workers=args.num_workers)

    model.eval()
    ious, fms = [], []

    for idx, (imgs, gt) in enumerate(tqdm(loader, total=len(loader))):
        imgs = imgs.unsqueeze(0).to(args.device)  # (1, T, 3, H, W)
        captions = [gt["caption"]]

        # --- 여기서 size와 frame_ids를 채워줘야 shape mismatch가 없어집니다 ---
        _, T, _, H, W = imgs.shape
        gt["frame_ids"] = list(range(T))
        gt["size"] = torch.as_tensor([H, W], device=imgs.device)

        outputs = model(imgs, captions, [gt])
        
        valid_idxs = torch.where(gt["valid"] == 1)[0]  # EDV, ESV 프레임
        # --- 출력 키를 pred_masks로 변경 ---
        pred_masks = outputs["pred_masks"].sigmoid().cpu()

        for vidx in valid_idxs:
            gmask = gt["masks"][vidx].numpy()
            pmask = (pred_masks[vidx] > args.threshold).numpy()

            # IoU(J) & Boundary F-measure(F) 계산
            iou = db_eval_iou(gmask[None], pmask[None]).mean()
            f = db_eval_boundary(gmask[None], pmask[None]).mean()
            ious.append(iou)
            fms.append(f)

            # (옵션) 시각화 저장
            if args.visualize:
                frame_img = (imgs[0, vidx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                frame_pil = Image.fromarray(frame_img).convert('RGB')  # RGBA 대신 RGB
                pmask_uint8 = (pmask * 255).astype(np.uint8)  # 마스크를 0~255 범위 uint8로 변환
                frame_vis = vis_add_mask(frame_pil, pmask_uint8, color_list[0])
                save_dir = os.path.join(save_visualize_path_prefix, f"sample_{idx}_frame_{vidx}.png")
                frame_vis.save(save_dir)

    # 최종 점수 계산
    J_score = np.mean(ious)
    F_score = np.mean(fms)
    JF_score = (J_score + F_score) / 2

    return J_score, F_score, JF_score


if __name__ == '__main__':
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser('SAMWISE EchoNet evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()

    name_exp = args.name_exp
    args.output_dir = os.path.join(args.output_dir, name_exp)

    main(args)
