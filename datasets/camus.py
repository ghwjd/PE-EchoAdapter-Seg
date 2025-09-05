from pathlib import Path
import random, json, os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools import mask as coco_mask
from datasets.transform_utils import FrameSampler, make_coco_transforms
from datasets.categories import camus_category_dict as category_dict


class CAMUSDataset(Dataset):
    """
    CAMUS Dataset Loader optimized:
    - 메타 생성 전 mask_dict 로드 및 필터링
    - sample_local_frames 사용으로 클립 내 연속 프레임 샘플링
    - 라벨은 exp 텍스트로 직접 매핑
    - engine.py, main.py, inference_camus.py와 호환
    """

    def __init__(self, img_folder: Path, ann_file: Path, transforms, num_frames: int):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.num_frames = num_frames

        # mask_dict 먼저 로드
        mask_json = img_folder / "mask_dict.json"
        print(f"Loading masks from {mask_json} ...")
        with open(mask_json, 'r') as fp:
            self.mask_dict = json.load(fp)

        # 메타데이터 생성 및 필터링
        self.prepare_metas()
        print(f"\nVideo count: {len(self.videos)}, Clip count: {len(self.metas)}\n")

    def prepare_metas(self):
        """meta_expressions.json을 읽고, 마스크가 하나라도 있는 클립만 저장"""
        with open(self.ann_file, 'r') as f:
            expr_data = json.load(f)["videos"]

        self.videos = list(expr_data.keys())
        self.metas = []
        for vid in self.videos:
            vid_data = expr_data[vid]
            frames = sorted(vid_data["frames"], key=lambda x: int(x))
            vid_len = len(frames)

            for exp_dict in vid_data["expressions"].values():
                obj_id = str(exp_dict["obj_id"])
                exp_text = " ".join(exp_dict["exp"].lower().split())

                # 각 클립 시작점별로 필터링
                for start in range(0, vid_len, self.num_frames):
                    clip = frames[start:start + self.num_frames]
                    # 해당 오브젝트가 하나라도 등장하는지 확인
                    valid = any(
                        self.mask_dict.get(vid, {}).get(f, {}).get(obj_id)
                        for f in clip
                    )
                    if not valid:
                        continue
                    self.metas.append({
                        'video': vid,
                        'exp': exp_text,
                        'obj_id': obj_id,
                        'frames': frames,
                        'frame_id': start
                    })

    @staticmethod
    def bounding_box(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        return torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.float32)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]
            vid, exp, obj_id, frames, start = (
                meta['video'], meta['exp'], meta['obj_id'],
                meta['frames'], meta['frame_id']
            )
            vid_len = len(frames)

            # 로컬 프레임 샘플링
            inds = FrameSampler.sample_local_frames(start, vid_len, self.num_frames)

            imgs, labels, boxes, masks, valids = [], [], [], [], []
            
            # 첫 번째 이미지에서 기준 크기 가져오기 (MeViS 패턴 참고)
            first_frame = frames[inds[0]]
            first_img_path = self.img_folder / "JPEGImages" / vid / f"{first_frame}.jpg"
            first_img = Image.open(first_img_path).convert("RGB")
            reference_size = first_img.size  # (w, h)
            
            for fi in inds:
                fname = frames[fi]
                img_path = self.img_folder / "JPEGImages" / vid / f"{fname}.jpg"
                img = Image.open(img_path).convert("RGB")
                
                # 모든 이미지를 기준 크기로 리사이즈 (EchoNet 패턴과 유사)
                if img.size != reference_size:
                    img = img.resize(reference_size, Image.BILINEAR)
                
                imgs.append(img)

                # RLE 디코딩 - MeViS 패턴 참고해서 numpy로 먼저 처리
                md = self.mask_dict.get(vid, {}).get(fname, {}).get(obj_id)
                if md:
                    mask = coco_mask.decode(md).squeeze().astype(np.float32)
                    # 마스크도 이미지와 동일한 크기로 조정
                    if mask.shape != (reference_size[1], reference_size[0]):  # (h, w)
                        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                        mask_img = mask_img.resize(reference_size, Image.NEAREST)
                        mask = np.array(mask_img).astype(np.float32) / 255.0
                else:
                    # 기준 크기로 빈 마스크 생성 (img.size[::-1]는 (h, w))
                    mask = np.zeros((reference_size[1], reference_size[0]), dtype=np.float32)
                
                # numpy에서 tensor로 변환 (EchoNet 패턴)
                mask = torch.from_numpy(mask)
                masks.append(mask)

                # bbox & valid - MeViS 패턴 참고
                if mask.any():
                    boxes.append(self.bounding_box(mask.numpy()))
                    valids.append(1)
                else:
                    boxes.append(torch.tensor([0, 0, 0, 0], dtype=torch.float32))
                    valids.append(0)

                # 라벨 매핑: 표현식 키로 바로 매핑 - MeViS에서는 category_id = 0 고정
                labels.append(torch.tensor(0, dtype=torch.int64))  # CAMUS도 0으로 통일

            # 타겟 구축 - MeViS/EchoNet 패턴 참고하되 engine.py 요구사항 반영
            w, h = reference_size
            boxes = torch.stack(boxes)
            boxes[:, 0::2].clamp_(0, w)
            boxes[:, 1::2].clamp_(0, h)
            labels = torch.stack(labels)
            masks = torch.stack(masks)  # 이제 모든 마스크가 같은 크기
            valids = torch.tensor(valids, dtype=torch.int64)

            target = {
                'frames_idx': torch.tensor(inds, dtype=torch.int64),
                'labels':     labels,
                'boxes':      boxes,
                'masks':      masks,
                'valid':      valids,
                'caption':    exp,
                'orig_size':  torch.tensor([h, w], dtype=torch.int64),
                'size':       torch.tensor([h, w], dtype=torch.int64),
                'video':      vid,  # engine.py에서 필요
                'video_id':   vid,  # MeViS 패턴 호환
                'exp_id':     idx,  # MeViS 패턴 호환
                'obj_id':     obj_id,
                'frames_str': frames, 
                'dataset_name': 'camus',  # evaluate 함수 호환
                'image_id':   torch.tensor(idx, dtype=torch.int64),  # evaluate 함수 호환
                'frame_ids':  inds,  # inference_camus.py 호환
            }

            imgs, target = self._transforms(imgs, target)
            imgs = torch.stack(imgs)
            target.update({
                'obj_id': obj_id,          # '1','2','3' 중 하나(문자열)
                'frames_str': frames,      # 원래 프레임 파일명 문자열 리스트 (['1','2',...])
            })

            # MeViS 패턴: valid한 인스턴스가 최소 1개 있는지 확인
            if torch.any(target['valid'] == 1):
                instance_check = True
            else:
                idx = random.randint(0, len(self) - 1)

        return imgs, target


def build(image_set, args):
    root = Path(args.camus_path)
    assert root.exists(), f"CAMUS path {root} not found"
    splits = {
        'train': ('train', 'meta_expressions.json'),
        'val':   ('val',   'meta_expressions.json'),
        'test':  ('test',  'meta_expressions.json'),
    }
    subdir, ann = splits[image_set]
    img_folder = root / subdir
    ann_file  = img_folder / ann
    return CAMUSDataset(
        img_folder, ann_file,
        transforms=make_coco_transforms(image_set, args.max_size, args.augm_resize),
        num_frames=args.num_frames
    )