# inference_camus.py
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, distributed as dist_data
from PIL import Image
from tqdm import tqdm

import util.misc as utils
import opts
from models.samwise import build_samwise
from util.misc import on_load_checkpoint
from tools.metrics import db_eval_boundary, db_eval_iou
from tools.colormap import colormap
from datasets.camus import CAMUSDataset
from datasets.transform_utils import make_coco_transforms, vis_add_mask

# colormap
color_list = colormap().astype('uint8').tolist()


def main(args):
    utils.init_distributed_mode(args)

    # Seed & cuDNN
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # I/O
    args.output_dir = os.path.join(args.output_dir, args.name_exp)
    os.makedirs(args.output_dir, exist_ok=True)
    args.log_file = os.path.join(args.output_dir, 'log.txt')
    with open(args.log_file, 'w') as fp:
        fp.writelines(" ".join(os.sys.argv) + '\n')
        fp.writelines(str(args.__dict__) + '\n\n')

    # Model
    device = torch.device(args.device)
    model = build_samwise(args).to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        if list(ckpt['model'].keys())[0].startswith('module'):
            ckpt['model'] = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
        ckpt = on_load_checkpoint(model_without_ddp, ckpt)
        miss, unexp = model_without_ddp.load_state_dict(ckpt['model'], strict=False)
        if utils.is_main_process():
            print(f'Missing Keys: {miss}')
            print(f'Unexpected Keys: {unexp}')

    if utils.is_main_process():
        print('Start CAMUS inference & evaluation')

    t0 = time.time()
    result = eval_camus(args, model, args.output_dir, split=args.split)
    # result = infer_frames_only(args, model, args.output_dir, split=args.split)
    if utils.is_main_process() and result is not None:
        J, F_, JF = result
        print(f'Final Evaluation Results → J: {J:.4f}, F: {F_:.4f}, J&F: {JF:.4f}')
        print(f"Total inference time: {time.time() - t0:.2f}s")


@torch.no_grad()

def eval_camus(args,
               model,
               out_dir,
               save_visualize_path_prefix=None,
               split='test'):
    
    if getattr(args, 'frames_only', False):
        return infer_frames_only(args, model, out_dir, split)
    """
    DDP 지원 + 환자/obj별 지표 저장
    """
    import json
    from collections import defaultdict
    import os

    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ==== Dataset ====
    root = Path(args.camus_path)
    ds = CAMUSDataset(
        img_folder=root / split,
        ann_file=(root / split / "meta_expressions.json"),
        transforms=make_coco_transforms(split, args.max_size, False),
        num_frames=args.num_frames,
    )

    ddp = bool(args.distributed and torch.distributed.is_initialized())
    if ddp:
        sampler = dist_data.DistributedSampler(ds, shuffle=False, drop_last=False)
    else:
        sampler = SequentialSampler(ds)

    loader = DataLoader(
        ds,
        sampler=sampler,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: batch[0],  # (imgs, target)
    )

    # vis path
    if save_visualize_path_prefix is None and args.visualize:
        save_visualize_path_prefix = os.path.join(out_dir, f'{split}_images', f'rank{utils.get_rank()}')
        os.makedirs(save_visualize_path_prefix, exist_ok=True)

    device = torch.device(args.device)
    model.eval()

    # 전체 평균 누적(프레임 단위)
    sum_iou = 0.0
    sum_f = 0.0
    cnt = 0.0

    # 환자/obj별 누적(샘플 단위 평균을 합산)
    acc = defaultdict(lambda: {"J": 0.0, "F": 0.0, "N": 0})

    it = tqdm(loader, total=len(loader), ncols=0) if utils.is_main_process() else loader

    for idx, (imgs, tgt) in enumerate(it):
        # imgs: (T,3,H,W) → (1,T,3,H,W)
        imgs = imgs.unsqueeze(0).to(device)

        # caption
        cap = tgt.get("caption", "left ventricle")
        captions = [cap]

        # forward 최소 타깃
        _, T, _, H, W = imgs.shape
        size = torch.as_tensor([H, W], device=device)
        frame_ids = tgt["frames_idx"].tolist() if isinstance(tgt["frames_idx"], torch.Tensor) else tgt["frames_idx"]
        target = {"size": size, "frame_ids": frame_ids}

        outputs = model(imgs, captions, [target])

        # pred_masks → (T,H,W)
        pred = outputs.get("pred_masks", None)
        if pred is None:
            pred = outputs["masks"]
        if pred.dim() == 5:       # [B,T,Q,H,W]
            pred = pred[0, :, 0]
        elif pred.dim() == 4:     # [B,T,H,W]
            pred = pred[0]
        elif pred.dim() == 3:     # [T,H,W]
            pred = pred
        else:
            raise RuntimeError(f"Unexpected pred shape: {tuple(pred.shape)}")

        # pmask = (torch.sigmoid(pred).cpu().numpy() > args.threshold).astype(np.uint8)
        pmask = (torch.sigmoid(pred).detach().cpu().numpy() > args.threshold).astype(np.uint8)


        # GT: (T,H,W)
        gtm = tgt["masks"].numpy().astype(np.uint8)

        # 평가 프레임 선택
        if "valid" in tgt:
            valid_flags = tgt["valid"].detach().cpu().numpy().astype(np.int64)
            eval_ids = np.where(valid_flags == 1)[0]
        else:
            # 백업: GT가 non-empty인 프레임만
            eval_ids = np.where(gtm.reshape(gtm.shape[0], -1).sum(axis=1) > 0)[0]

        # --- 한 샘플(=한 환자·한 obj·한 클립) 내부 평균 ---
        sample_J_sum, sample_F_sum, sample_cnt = 0.0, 0.0, 0

        for t in eval_ids:
            gi = gtm[t]
            pi = pmask[t]
            iou = float(db_eval_iou(gi[None], pi[None]).mean())
            f = float(db_eval_boundary(gi[None], pi[None]).mean())

            # 전역 프레임 단위 누적
            sum_iou += iou
            sum_f += f
            cnt += 1.0

            # 샘플 내부 누적
            sample_J_sum += iou
            sample_F_sum += f
            sample_cnt += 1

            # ---- 시각화 (회색 원본 + 오버레이) ----
            if args.visualize and save_visualize_path_prefix is not None:
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img = imgs[0, t].detach().cpu().numpy().transpose(1, 2, 0)
                img = (img * std + mean)
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                gray = np.mean(img, axis=2).astype(np.uint8)
                frame_pil = Image.fromarray(gray).convert("RGBA")

                vid = str(tgt['video'])
                obj = str(tgt.get('obj_id', '?'))                 # ← CAMUSDataset에서 넣어줄 것(아래 참고)
                exp = str(tgt.get('caption', '')).replace(' ','_')

                # frame name 복원(없으면 인덱스)
                if isinstance(tgt['frames_idx'], torch.Tensor):
                    frame_ids_list = tgt['frames_idx'].detach().cpu().tolist()
                else:
                    frame_ids_list = list(tgt['frames_idx'])
                if 'frames_str' in tgt:
                    frame_str_list = tgt['frames_str']
                    frame_str = str(frame_str_list[t])
                else:
                    frame_str = str(frame_ids_list[t])

                vis_dir = os.path.join(save_visualize_path_prefix, vid, f"obj{obj}_{exp}")
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"f{frame_str}.png")

                vis_img = vis_add_mask(frame_pil, pi, color_list[0])
                vis_img.save(vis_path)

        # --- 샘플 평균을 환자/obj 키로 합산 ---
        if sample_cnt > 0:
            vid = str(tgt["video"])
            obj = str(tgt.get("obj_id", "?"))
            acc[(vid, obj)]["J"] += (sample_J_sum / sample_cnt)
            acc[(vid, obj)]["F"] += (sample_F_sum / sample_cnt)
            acc[(vid, obj)]["N"] += 1

    # ==== DDP: 환자/obj 누적 합치기 ====
    rank_acc = dict(acc)
    if ddp:
        obj_list = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(obj_list, rank_acc)
        merged = defaultdict(lambda: {"J": 0.0, "F": 0.0, "N": 0})
        for d in obj_list:
            for k, v in d.items():
                merged[k]["J"] += v["J"]; merged[k]["F"] += v["F"]; merged[k]["N"] += v["N"]
    else:
        merged = rank_acc

    # ==== 환자/obj별 최종 평균 → JSON 저장 ====
    if utils.is_main_process():
        per_patient = {}
        for (vid, obj), v in merged.items():
            if v["N"] == 0: 
                continue
            Jm = v["J"] / v["N"]
            Fm = v["F"] / v["N"]
            JFm = 0.5 * (Jm + Fm)
            per_patient.setdefault(vid, {})[obj] = {"J": float(Jm), "F": float(Fm), "JF": float(JFm)}

        # (선택) 환자 단일 스칼라(모든 obj 평균)도 추가
        patient_mean = {}
        for vid, objs in per_patient.items():
            vals = [o["JF"] for o in objs.values()]
            if len(vals):
                patient_mean[vid] = float(np.mean(vals))

        save_json = os.path.join(out_dir, f"{split}_per_patient_metrics.json")
        with open(save_json, "w") as f:
            json.dump({"per_patient": per_patient, "patient_mean": patient_mean}, f, indent=2)
        print(f"[✓] saved per-patient metrics → {save_json}")

    # ==== 전역 평균 (프레임 단위) ====
    totals = torch.tensor([sum_iou, sum_f, cnt], dtype=torch.float64, device=device)
    if ddp:
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    total_iou, total_f, total_cnt = totals.tolist()
    if total_cnt == 0:
        J, F_, JF = 0.0, 0.0, 0.0
    else:
        J = total_iou / total_cnt
        F_ = total_f / total_cnt
        JF = 0.5 * (J + F_)

    if utils.is_main_process():
        return (J, F_, JF)
    else:
        return None

import glob
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import os

def infer_frames_only(args, model, out_dir, split='test'):
    root = Path(args.camus_path) / split / "JPEGImages"
    assert root.exists(), f"Not found: {root}"

    # ✅ out_dir는 이미 name_exp까지 포함되어 있음
    save_root = out_dir
    os.makedirs(save_root, exist_ok=True)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    device = torch.device(args.device)
    model.eval()

    videos = sorted([p for p in root.iterdir() if p.is_dir()])
    total_saved = 0

    for vdir in videos:
        frame_files = sorted(
            [p for p in vdir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')],
            key=lambda p: int(''.join([c for c in p.stem if c.isdigit()]) or -1)
        )
        if len(frame_files) == 0:
            continue

        out_dir_video = os.path.join(save_root, vdir.name)
        os.makedirs(out_dir_video, exist_ok=True)

        T = args.num_frames
        for s in range(0, len(frame_files), T):
            clip = frame_files[s:s+T]

            ref_img = Image.open(clip[0]).convert('RGB')
            ref_size = ref_img.size  # (W,H)

            imgs_np = []
            for fpath in clip:
                im = Image.open(fpath).convert('RGB')
                if im.size != ref_size:
                    im = im.resize(ref_size, Image.BILINEAR)
                arr = np.asarray(im).astype(np.float32) / 255.0
                arr = (arr - mean) / std
                arr = arr.transpose(2, 0, 1)  # CHW
                imgs_np.append(arr)
            imgs = torch.from_numpy(np.stack(imgs_np, axis=0)).unsqueeze(0).to(device)  # (1,T,3,H,W)

            _, T_, C, H, W = imgs.shape
            size = torch.as_tensor([H, W], device=device)
            frame_ids = list(range(T_))
            target = {"size": size, "frame_ids": frame_ids}
            captions = [getattr(args, 'caption', 'left ventricle endocardium')]

            # ✅ 추론은 no_grad로
            with torch.no_grad():
                outputs = model(imgs, captions, [target])
                pred = outputs.get("pred_masks", None)
                if pred is None:
                    pred = outputs.get("masks", None)
                if pred is None:
                    raise KeyError(f"Model outputs has neither 'pred_masks' nor 'masks' keys: {list(outputs.keys())}")

                if pred.dim() == 5:   # [B,T,Q,H,W]
                    pred = pred[0, :, 0]
                elif pred.dim() == 4: # [B,T,H,W]
                    pred = pred[0]
                elif pred.dim() == 3: # [T,H,W]
                    pass
                else:
                    raise RuntimeError(f"Unexpected pred shape: {tuple(pred.shape)}")

                # ✅ detach 후 numpy
                pmask = (torch.sigmoid(pred).detach().cpu().numpy() > args.threshold).astype(np.uint8)

            # 저장: <save_root>/<video>/<frame:03d>.png (0/255 단일채널)
            for k, fpath in enumerate(clip):
                try:
                    fid = int(''.join([c for c in fpath.stem if c.isdigit()]))
                except ValueError:
                    fid = s + k
                out_name = f"{fid:03d}.png"
                out_path = os.path.join(out_dir_video, out_name)

                m = (pmask[k].astype(np.uint8) * 255)
                Image.fromarray(m, mode='L').save(out_path)
                total_saved += 1

    if utils.is_main_process():
        print(f"[✓] frames_only: saved {total_saved} masks to {save_root}")
    return None



if __name__ == '__main__':
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser('SAMWISE CAMUS evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    main(args)
