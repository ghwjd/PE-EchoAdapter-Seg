# datasets/echonet.py
from pathlib import Path
import random, json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools import mask as coco_mask
from datasets.transform_utils import FrameSampler, make_coco_transforms
from datasets.categories import echonet_category_dict as category_dict

class EchoNetDataset(Dataset):
    """
    EchoNet-Dynamic 로더:
    - mask_dict.json을 RLE로 읽고 decode
    - meta_expressions.json에서 'left ventricle'과 'LV' 중 랜덤으로 하나의 프롬프트 선택
    - echonet_category_dict에서 라벨 ID mapping
    """
    def __init__(self, split_dir: Path, transforms, num_frames: int):
        self.img_root   = split_dir / "JPEGImages"
        self.expr_file  = split_dir / "meta_expressions.json"
        self.mask_file  = split_dir / "mask_dict.json"
        self._tfm       = transforms
        self.num_frames = num_frames

        # meta_expressions.json 읽기
        with open(self.expr_file, 'r', encoding='utf-8') as f:
            expr = json.load(f)["videos"]
        self.videos = list(expr.keys())
        self.expr_by_video = expr

        # mask_dict.json 읽기
        with open(self.mask_file, 'r', encoding='utf-8') as f:
            self.mask_dict = json.load(f)

        # clip 메타 준비 (exp는 __getitem__에서 랜덤 선택)
        self.metas = []
        for vid in self.videos:
            frames = sorted(self.expr_by_video[vid]["frames"], key=lambda x: int(x))
            for start in range(0, len(frames), num_frames):
                self.metas.append({"video": vid, "frames": frames, "frame_id": start})

    @staticmethod
    def bbox_from_mask(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return torch.tensor([0, 0, 0, 0], dtype=torch.float32), 0
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        return torch.tensor([x1, y1, x2, y2], dtype=torch.float32), 1

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        while True:
            m = self.metas[idx]
            vid, frames, start = m["video"], m["frames"], m["frame_id"]

            # expressions에 담긴 여러 프롬프트 중 하나를 랜덤으로 선택
            exp_entries = self.expr_by_video[vid]["expressions"]
            exp = random.choice(list(exp_entries.values()))["exp"]

            inds = FrameSampler.sample_local_frames(start, len(frames), self.num_frames)
            imgs, masks, boxes, labels, valids = [], [], [], [], []

            for fi in inds:
                fname = frames[fi]
                img = Image.open(self.img_root / vid / f"{fname}.jpg").convert("RGB")
                imgs.append(img)

                # mask RLE 디코딩 → 바로 tensor로 변환
                md = self.mask_dict.get(vid, {}).get(str(fname), {}).get("1")
                if md:
                    mask = coco_mask.decode(md).squeeze().astype(np.uint8)
                    mask = torch.from_numpy(mask)  # ✅ 바로 tensor로 변환
                else:
                    mask = torch.zeros(img.size[::-1], dtype=torch.uint8)  # ✅ tensor로 생성
                masks.append(mask)

                # bbox, valid flag
                box, v = self.bbox_from_mask(mask.numpy() if isinstance(mask, torch.Tensor) else mask)
                boxes.append(box)
                valids.append(v)

                # category ID: echonet_category_dict['left ventricle']==0
                labels.append(torch.tensor(category_dict["left ventricle"], dtype=torch.int64))

            # build target
            w, h = imgs[0].size
            target = {
                "frames_idx": torch.tensor(inds, dtype=torch.int64),
                "labels":     torch.stack(labels),
                "boxes":      torch.stack(boxes),
                "masks":      torch.stack(masks),  # ✅ 이제 모두 tensor
                "valid":      torch.tensor(valids, dtype=torch.int64),
                "caption":    " ".join(exp.lower().split()),
                "orig_size":  torch.tensor([h, w], dtype=torch.int64),
                "size":       torch.tensor([h, w], dtype=torch.int64),
                "video":      vid,
            }

            # transform & stack
            imgs, target = self._tfm(imgs, target)
            imgs = torch.stack(imgs)

            if torch.any(target["valid"] == 1):
                return imgs, target
            idx = random.randint(0, len(self) - 1)

def build(image_set: str, args):
    root = Path(args.echonet_path) / (
        "train" if image_set == "train" else ("valid" if image_set == "valid" else "test")
    )
    assert root.exists(), f"{root} not found"
    return EchoNetDataset(
        split_dir=root,
        transforms=make_coco_transforms(image_set, args.max_size, args.augm_resize),
        num_frames=args.num_frames
    )