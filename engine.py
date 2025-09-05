"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import math
import os
import sys
from typing import Iterable
import json # *
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tools.metrics import calculate_precision_at_k_and_iou_metrics
import util.misc as utils
from torch.nn import functional as F
from models.segmentation import loss_masks
from pathlib import Path


def _to_py(obj): # *
    """torch.Tensor 등을 JSON 직렬화 가능하게 변환"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_py(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    return obj

def _extract_sample_meta(targets): # *
    """
    targets 리스트에서 가능한 메타만 뽑아오기
    EchoNetDataset가 video / frames_idx 등을 제공하면 포함, 없으면 가능한 키만.
    """
    meta = []
    for t in targets:
        m = {}
        # 자주 쓰는 키들만 골라 안전하게 추출
        for key in ["video", "video_id", "frames_idx", "caption", "orig_size", "size"]:
            if key in t:
                m[key] = _to_py(t[key])
        meta.append(m)
    return meta

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    lr_scheduler=None, args=None, writer=None, log_freq: int = 50): # ★ writer, log_freq추가
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    warmup_steps = 500
    warmup_factor = 1. / warmup_steps
    print_freq = 50

    # ===== 스파이크 로깅 설정 =====
    spike_path = getattr(args, "spike_log_path", None)
    if spike_path is not None:
        spike_path = Path(spike_path)
    loss_spike_thresh = getattr(args, "loss_spike_thresh", 0.5)
    cme_spike_thresh  = getattr(args, "cme_spike_thresh",  0.5)
    # ===========================
    
    step=0
    num_steps_per_epoch = len(data_loader) if hasattr(data_loader, "__len__") else None

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        step+=1
        
        # # --- [수정] Warmup 로직 적용 ---
        global_step = epoch * num_steps_per_epoch + step if num_steps_per_epoch is not None else step
        if epoch == 0 and global_step < warmup_steps:
            lr_scale = global_step * warmup_factor
            for group in optimizer.param_groups:
                # 각 파라미터 그룹의 원래 목표 lr에 스케일을 곱해줍니다.
                group['lr'] = group['initial_lr'] * lr_scale
        # # -----------------------------
        
        model.train()
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        outputs = model(samples, captions, targets)
        losses = {}
        seg_loss = loss_masks(torch.cat(outputs["masks"]), targets, num_frames=samples.tensors.shape[1])
        losses.update(seg_loss)
        if args.use_cme_head and "pred_cme_logits" in outputs:
            weight = torch.tensor([1., 2.]).to(device)
            CME_loss = F.cross_entropy(torch.cat(outputs["pred_cme_logits"]), ignore_index=-1,
                                        target=torch.tensor(outputs["cme_label"]).long().to(device),
                                        weight=weight)
            losses.update({"CME_loss": CME_loss if not CME_loss.isnan() else torch.tensor(0).to(device)})

        loss_dict = losses
        losses = sum(loss_dict[k] for k in loss_dict.keys())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v for k, v in loss_dict_reduced.items()}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        optimizer.step()
        # # [수정] Warmup 기간에는 lr_scheduler.step()을 호출하지 않습니다.
        if epoch > 0 or global_step >= warmup_steps:
             if lr_scheduler is not None:
                lr_scheduler.step()
        # ---------------------------------------------------------
        # lr_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        
        # ----- TensorBoard 간단 기록 -----
        if writer is not None and utils.is_main_process():
            # 스텝 인덱스(글로벌) — DataLoader 길이를 알면 예쁘게, 아니면 누적 step 사용
            if num_steps_per_epoch is not None:
                global_step = epoch * num_steps_per_epoch + (step - 1)
            else:
                global_step = step
            if step % log_freq == 0:
                writer.add_scalar('iter/loss', loss_value, global_step)
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar(f'iter/{k}', v.item() if hasattr(v, "item") else float(v), global_step)
                writer.add_scalar('iter/lr', optimizer.param_groups[0]['lr'], global_step)
                
                # 게이팅 파라미터 값 텐서보드에 기록
                if args.use_sam_da_adapter:
                    for i, layer in enumerate(model.module.sam.sam_mask_decoder.transformer.layers):
                        if hasattr(layer, 'adapter'): # 어댑터가 존재하는지 안전하게 확인
                            # gating_value = layer.adapter.gating_param.item()
                            g_param = layer.adapter.gating_param
                            if g_param.numel() > 1:
                                gating_value = g_param.detach().mean().item()
                                # gating_vector = [round(x, 4) for x in g_param.detach().cpu().tolist()]
                                # logger.log({'gating_vector': gating_vector})
                            else:
                                gating_value = g_param.item()
                            writer.add_scalar(f'adapter/layer_{i}_gating_param', gating_value, global_step)
                # -------------------

        # ----- 스파이크 탐지 & 기록 (JSONL) -----
        if spike_path is not None and utils.is_main_process():
            loss_dice_val = float(loss_dict_reduced.get("loss_dice", torch.tensor(0.)).item())
            cme_val = float(loss_dict_reduced.get("CME_loss", torch.tensor(0.)).item())
            is_spike = (loss_value >= loss_spike_thresh) or (cme_val >= cme_spike_thresh)

            if is_spike:
                try:
                    record = {
                        "epoch": epoch,
                        "step_in_epoch": step,
                        "global_step": (epoch * num_steps_per_epoch + (step - 1)) if num_steps_per_epoch is not None else step,
                        "loss": loss_value,
                        "loss_dice": loss_dice_val,
                        "CME_loss": cme_val,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "batch_meta": _extract_sample_meta(targets),
                    }
                    spike_path.parent.mkdir(parents=True, exist_ok=True)
                    with spike_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"[warn] spike logging failed: {e}")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, postprocessors, data_loader, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    predictions = []
    for samples, targets in metric_logger.log_every(data_loader, 20, header):
        dataset_name = targets[0]["dataset_name"]
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        targets = utils.targets_to(targets, device)

        outputs = model(samples, captions, targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm']([], outputs, orig_target_sizes, target_sizes)

        # REC & RES predictions
        for p, target in zip(results, targets):
            for m in p['rle_masks']:
                predictions.append({'image_id': target['image_id'].item(),
                                    'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                    'segmentation': m,
                                    'score': 1
                                    })

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # evaluate RES
    # gather and merge predictions from all gpus
    gathered_pred_lists = utils.all_gather(predictions)
    predictions = [p for p_list in gathered_pred_lists for p in p_list]

    eval_metrics = {}
    if utils.is_main_process():
        if dataset_name == 'refcoco':
            coco_gt = COCO(os.path.join(args.coco_path, 'refcoco/instances_refcoco_val.json'))
        elif dataset_name == 'refcoco+':
            coco_gt = COCO(os.path.join(args.coco_path, 'refcoco+/instances_refcoco+_val.json'))
        elif dataset_name == 'refcocog':
            coco_gt = COCO(os.path.join(args.coco_path, 'refcocog/instances_refcocog_val.json'))
        else:
            raise NotImplementedError
        coco_pred = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
        coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
        ap_metrics = coco_eval.stats[:6]
        eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
        # Precision and IOU
        precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
        eval_metrics.update({f'segm P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
        eval_metrics.update({'segm overall_iou': overall_iou, 'segm mean_iou': mean_iou})
        print(eval_metrics)

    return eval_metrics
