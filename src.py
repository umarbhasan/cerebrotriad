"""
SPMT-Net: Similarity-Preserving Multi-Task Knowledge Distillation
for efficient joint brain-tumor segmentation and classification on BRISC-2025.

Single-GPU, Kaggle-ready. No DataParallel / DistributedDataParallel anywhere.
Includes native resumption, FP16 overflow protection, and strict VRAM management.

Pipeline
--------
Teacher  : strong timm encoder + U-Net decoder + multi-scale classification head
Student  : compact timm encoder (default MobileNetV3-Large) + light U-Net decoder + cls head
Transfer : (1) task losses (Dice+BCE for seg, CE for cls)
           (2) response/logit KD (per-pixel for seg, temperature KL for cls)
           (3) canonical Similarity-Preserving KD (Tung & Mori, ICCV'19) on the
               deepest encoder stages + the pooled classification feature.
"""

from __future__ import annotations
import os, sys, json, time, random, warnings, math, gc, copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# 0. Configuration
# ------------------------------------------------------------------ #
@dataclass
class Config:
    # Reproducibility
    seed: int = 42

    # Data paths (auto-resolved at load time)
    data_root: str = "/kaggle/input/datasets/briscdataset/brisc2025/brisc2025/segmentation_task"
    cls_data_root: str = "/kaggle/input/datasets/briscdataset/brisc2025/brisc2025/classification_task"
    img_size: int = 256
    val_split: float = 0.20
    
    class_map: Dict[str, int] = field(default_factory=lambda: {"gl": 0, "me": 1, "pi": 2})
    class_names: Tuple[str, ...] = ("glioma", "meningioma", "pituitary")
    cls_class_names: Tuple[str, ...] = ("glioma", "meningioma", "no_tumor", "pituitary")

    # Encoders
    teacher_encoder: str = "convnext_small"
    student_encoder: str = "mobilenetv3_large_100"
    pretrained: bool = True
    decoder_channels_teacher: Tuple[int, ...] = (256, 128, 64, 32)
    decoder_channels_student: Tuple[int, ...] = (128, 64, 32, 16)
    n_spkd_stages: int = 3

    # Optimisation
    batch_size: int = 16
    epochs_teacher: int = 80
    epochs_student: int = 80
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    grad_clip: float = 5.0
    patience: int = 20
    amp: bool = True  # Set to False for standalone classification to ensure stability
    num_workers: int = 2

    # Multi-task weights
    w_seg: float = 1.0
    w_cls: float = 0.5

    # Distillation weights (student only)
    kd_seg: float = 1.0
    kd_cls: float = 1.0
    kd_temp: float = 4.0
    spkd_w: float = 50.0

    # IO
    out_dir: str = "/kaggle/working/spmt_out"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def dump(self):
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "config.json"), "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in asdict(self).items()}, f, indent=2)


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ------------------------------------------------------------------ #
# 1. Data Processing
# ------------------------------------------------------------------ #
def _resolve_data_dir(configured: str, marker: str) -> str:
    if os.path.isdir(configured):
        return configured
    import glob
    for base in ("/kaggle/input", "/kaggle/working", "."):
        hits = glob.glob(os.path.join(base, "**", marker), recursive=True)
        hits = [h for h in hits if os.path.isdir(h)]
        if hits:
            print(f"[data] '{configured}' not found; using '{hits[0]}'")
            return hits[0]
    raise FileNotFoundError(f"Could not find '{marker}'.")


def parse_class_from_name(filename: str, class_map: Dict[str, int]) -> int:
    stem = Path(filename).stem
    toks = stem.split("_")
    for t in toks:
        if t in class_map:
            return class_map[t]
    for code, idx in class_map.items():
        if f"_{code}_" in stem:
            return idx
    raise ValueError(f"Could not parse tumor code from '{filename}'")


def list_pairs(split_dir: str, class_map: Dict[str, int]):
    img_dir, msk_dir = os.path.join(split_dir, "images"), os.path.join(split_dir, "masks")
    imgs = sorted([str(p) for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
                   for p in Path(img_dir).glob(e)])
    images, masks, labels = [], [], []
    for ip in imgs:
        stem = Path(ip).stem
        mp = None
        for e in ("png", "jpg", "jpeg", "bmp"):
            cand = os.path.join(msk_dir, f"{stem}.{e}")
            if os.path.exists(cand):
                mp = cand; break
        if mp is None: continue
        images.append(ip); masks.append(mp)
        labels.append(parse_class_from_name(stem, class_map))
    return images, masks, labels


def list_classification(split_dir: str, class_names: Tuple[str, ...]):
    paths, labels = [], []
    for idx, cname in enumerate(class_names):
        cdir = os.path.join(split_dir, cname)
        if not os.path.isdir(cdir): continue
        for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for p in Path(cdir).glob(e):
                paths.append(str(p)); labels.append(idx)
    return paths, labels


def build_transforms(img_size: int):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.4),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.RandomGamma(gamma_limit=(85, 115), p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    eval_ = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    return train, eval_


class BRISCMultiTask(torch.utils.data.Dataset):
    def __init__(self, images, masks, labels, tfm):
        self.images, self.masks, self.labels, self.tfm = images, masks, labels, tfm

    def __len__(self): return len(self.images)

    def __getitem__(self, i):
        import cv2
        img = cv2.cvtColor(cv2.imread(self.images[i]), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)
        msk = (msk > 127).astype(np.float32)
        out = self.tfm(image=img, mask=msk)
        image, mask = out["image"], out["mask"]
        if mask.ndim == 2: mask = mask.unsqueeze(0)
        return {"image": image, "mask": mask.float(),
                "label": torch.tensor(self.labels[i], dtype=torch.long)}


class BRISCClassification(torch.utils.data.Dataset):
    def __init__(self, paths, labels, tfm):
        self.paths, self.labels, self.tfm = paths, labels, tfm

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        import cv2
        img = cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB)
        image = self.tfm(image=img)["image"]
        return {"image": image, "label": torch.tensor(self.labels[i], dtype=torch.long)}


def make_loaders(cfg: Config):
    from sklearn.model_selection import train_test_split
    root = _resolve_data_dir(cfg.data_root, "segmentation_task")
    tr_img, tr_msk, tr_lbl = list_pairs(os.path.join(root, "train"), cfg.class_map)
    te_img, te_msk, te_lbl = list_pairs(os.path.join(root, "test"), cfg.class_map)
    xi, vi, xm, vm, xl, vl = train_test_split(
        tr_img, tr_msk, tr_lbl, test_size=cfg.val_split,
        random_state=cfg.seed, shuffle=True, stratify=tr_lbl)
    t_tfm, e_tfm = build_transforms(cfg.img_size)
    dl = lambda ds, sh, dl_: torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=sh, num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"), drop_last=dl_)
    return (dl(BRISCMultiTask(xi, xm, xl, t_tfm), True, True),
            dl(BRISCMultiTask(vi, vm, vl, e_tfm), False, False),
            dl(BRISCMultiTask(te_img, te_msk, te_lbl, e_tfm), False, False),
            np.bincount(te_lbl, minlength=len(cfg.class_names)))


def make_cls_loaders(cfg: Config):
    from sklearn.model_selection import train_test_split
    root = _resolve_data_dir(cfg.cls_data_root, "classification_task")
    tr_p, tr_l = list_classification(os.path.join(root, "train"), cfg.cls_class_names)
    te_p, te_l = list_classification(os.path.join(root, "test"), cfg.cls_class_names)
    xp, vp, xl, vl = train_test_split(tr_p, tr_l, test_size=cfg.val_split,
                                      random_state=cfg.seed, shuffle=True, stratify=tr_l)
    t_tfm, e_tfm = build_transforms(cfg.img_size)
    dl = lambda ds, sh, dl_: torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=sh, num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"), drop_last=dl_)
    return (dl(BRISCClassification(xp, xl, t_tfm), True, True),
            dl(BRISCClassification(vp, vl, e_tfm), False, False),
            dl(BRISCClassification(te_p, te_l, e_tfm), False, False))


# ------------------------------------------------------------------ #
# 2. Model Architecture
# ------------------------------------------------------------------ #
def _to_nchw(t: torch.Tensor, c: int) -> torch.Tensor:
    if t.dim() == 4 and t.shape[1] != c and t.shape[-1] == c:
        t = t.permute(0, 3, 1, 2).contiguous()
    return t

class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, p=None, act=True):
        p = k // 2 if p is None else p
        layers = [nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout)]
        if act: layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.reduce = ConvBNAct(in_c, out_c, k=1)
        self.fuse = nn.Sequential(ConvBNAct(out_c + skip_c, out_c, k=3),
                                  ConvBNAct(out_c, out_c, k=3))

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.reduce(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.fuse(x)

class MultiTaskNet(nn.Module):
    def __init__(self, encoder_name: str, num_classes: int,
                 decoder_channels: Tuple[int, ...], pretrained: bool = True,
                 aux_seg: bool = False):
        super().__init__()
        import timm
        self.encoder = timm.create_model(encoder_name, features_only=True, pretrained=pretrained)
        self.enc_channels = list(self.encoder.feature_info.channels())
        self.n_stages = len(self.enc_channels)
        self.aux_seg = aux_seg

        dec_c = list(decoder_channels)
        n_steps = self.n_stages - 1
        while len(dec_c) < n_steps: dec_c.append(dec_c[-1])
        dec_c = dec_c[:n_steps]

        self.dec_blocks = nn.ModuleList()
        in_c = self.enc_channels[-1]
        rev_skip = self.enc_channels[:-1][::-1]
        for i in range(n_steps):
            out_c = dec_c[i]
            self.dec_blocks.append(UNetDecoderBlock(in_c, rev_skip[i], out_c))
            in_c = out_c
            
        self.seg_head = nn.Sequential(ConvBNAct(in_c, in_c, k=3), nn.Conv2d(in_c, 1, kernel_size=1))
        if aux_seg:
            self.aux_head = nn.Conv2d(self.enc_channels[-1], 1, kernel_size=1)

        self.cls_taps = list(range(self.n_stages))[-min(3, self.n_stages):]
        cls_in = sum(self.enc_channels[t] for t in self.cls_taps)
        self.cls_norm = nn.LayerNorm(cls_in)
        self.cls_head = nn.Sequential(
            nn.Linear(cls_in, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, num_classes))
        self._cls_in = cls_in

    def forward(self, x, compute_seg=True):
        H, W = x.shape[-2:]
        raw = self.encoder(x)
        feats = [_to_nchw(f, c) for f, c in zip(raw, self.enc_channels)]

        seg = None
        if compute_seg:
            d = feats[-1]
            skips = feats[:-1][::-1]
            for i, blk in enumerate(self.dec_blocks):
                d = blk(d, skips[i])
            seg = F.interpolate(self.seg_head(d), size=(H, W), mode="bilinear", align_corners=False)

        pooled = [F.adaptive_avg_pool2d(feats[t], 1).flatten(1) for t in self.cls_taps]
        cls_feat = torch.cat(pooled, dim=1)
        cls = self.cls_head(self.cls_norm(cls_feat))

        out = {"seg": seg, "cls": cls, "feats": feats, "cls_feat": cls_feat}
        if compute_seg and self.aux_seg:
            out["aux"] = F.interpolate(self.aux_head(feats[-1]), size=(H, W),
                                       mode="bilinear", align_corners=False)
        return out


def build_model(role: str, cfg: Config, num_classes: int, pretrained: Optional[bool] = None):
    pretrained = cfg.pretrained if pretrained is None else pretrained
    if role == "teacher":
        return MultiTaskNet(cfg.teacher_encoder, num_classes, cfg.decoder_channels_teacher, pretrained, aux_seg=False)
    return MultiTaskNet(cfg.student_encoder, num_classes, cfg.decoder_channels_student, pretrained, aux_seg=True)


# ------------------------------------------------------------------ #
# 3. Losses & Distillation Core
# ------------------------------------------------------------------ #
def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    p = p.reshape(p.size(0), -1)
    t = target.reshape(target.size(0), -1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()

def seg_loss(logits, target):
    return dice_loss(logits, target) + F.binary_cross_entropy_with_logits(logits, target)

def spkd_pair_stable(fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
    b = fs.size(0)
    qs = fs.reshape(b, -1).float()
    qt = ft.reshape(b, -1).float()
    Gs = qs @ qs.t()
    Gt = qt @ qt.t()
    # Explicit norm with clamp_min prevents NaN gradients from zero-vectors
    gs = Gs / Gs.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
    gt = Gt / Gt.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
    return (gs - gt).pow(2).sum() / (b * b)


class DistillLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.c = cfg

    def forward(self, s_out, t_out, mask, label):
        c = self.c
        L = {}
        L["seg"] = seg_loss(s_out["seg"], mask)
        L["cls"] = F.cross_entropy(s_out["cls"], label)
        task = c.w_seg * L["seg"] + c.w_cls * L["cls"]
        
        if "aux" in s_out:
            L["aux"] = seg_loss(s_out["aux"], mask)
            task = task + 0.4 * L["aux"]

        total = task
        if t_out is not None:
            L["kd_seg"] = F.mse_loss(torch.sigmoid(s_out["seg"]), torch.sigmoid(t_out["seg"]))
            T = c.kd_temp
            # Cast to FP32 before scaling to avoid FP16 overflow
            L["kd_cls"] = F.kl_div(
                F.log_softmax(s_out["cls"].float() / T, 1),
                F.softmax(t_out["cls"].float() / T, 1),
                reduction="batchmean"
            ) * (T * T)
            
            sp = 0.0
            k = c.n_spkd_stages
            for fs, ft in zip(s_out["feats"][-k:], t_out["feats"][-k:]):
                sp = sp + spkd_pair_stable(fs, ft)
            sp = sp + spkd_pair_stable(s_out["cls_feat"], t_out["cls_feat"])
            L["spkd"] = sp
            
            total = total + c.kd_seg * L["kd_seg"] + c.kd_cls * L["kd_cls"] + c.spkd_w * L["spkd"]
            
        L["total"] = total
        return total, {k_: (v.item() if torch.is_tensor(v) else float(v)) for k_, v in L.items()}


class ClsDistillLossStable(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.c = cfg

    def forward(self, s_out, t_out, label):
        c = self.c
        L = {"cls": F.cross_entropy(s_out["cls"].float(), label)}
        total = L["cls"]
        
        if t_out is not None:
            T = c.kd_temp
            L["kd_cls"] = F.kl_div(
                F.log_softmax(s_out["cls"].float() / T, 1),
                F.softmax(t_out["cls"].float() / T, 1),
                reduction="batchmean"
            ) * (T * T)
            
            sp = 0.0
            for fs, ft in zip(s_out["feats"][-c.n_spkd_stages:], t_out["feats"][-c.n_spkd_stages:]):
                sp = sp + spkd_pair_stable(fs, ft)
            sp = sp + spkd_pair_stable(s_out["cls_feat"], t_out["cls_feat"])
            L["spkd"] = sp
            total = total + c.kd_cls * L["kd_cls"] + c.spkd_w * L["spkd"]
            
        L["total"] = total
        return total, {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in L.items()}


# ------------------------------------------------------------------ #
# 4. Evaluation & Metrics
# ------------------------------------------------------------------ #
@torch.no_grad()
def seg_scores(logits, mask, thr=0.5, eps=1e-6):
    p = (torch.sigmoid(logits) > thr).float().reshape(logits.size(0), -1)
    t = (mask > 0.5).float().reshape(mask.size(0), -1)
    inter = (p * t).sum(1)
    union = p.sum(1) + t.sum(1) - inter
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)
    return iou.cpu().numpy(), dice.cpu().numpy()

def summarize_seg(ious, dices, labels, class_names, test_counts):
    labels = np.asarray(labels)
    per_class_iou, per_class_dice = [], []
    for ci in range(len(class_names)):
        m = labels == ci
        per_class_iou.append(float(np.mean(ious[m])) if m.any() else float("nan"))
        per_class_dice.append(float(np.mean(dices[m])) if m.any() else float("nan"))
    w = np.asarray(test_counts, dtype=float); w = w / w.sum()
    return {
        "per_class_iou": {class_names[i]: per_class_iou[i] for i in range(len(class_names))},
        "per_class_dice": {class_names[i]: per_class_dice[i] for i in range(len(class_names))},
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "weighted_mIoU": float(np.nansum(np.array(per_class_iou) * w)),
    }

def summarize_cls(y_true, y_pred, class_names):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    acc = accuracy_score(y_true, y_pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(acc),
        "macro": {"precision": float(p_m), "recall": float(r_m), "f1": float(f_m)},
        "weighted": {"precision": float(p_w), "recall": float(r_w), "f1": float(f_w)},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def measure_flops(model, cfg: Config):
    x = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=cfg.device)
    model.eval()
    try:
        from fvcore.nn import FlopCountAnalysis
        return float(FlopCountAnalysis(model, x).total()) / 1e9
    except Exception: pass
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(model, (3, cfg.img_size, cfg.img_size), as_strings=False, print_per_layer_stat=False, verbose=False)
        return float(macs) * 2 / 1e9
    except Exception: return None

@torch.no_grad()
def measure_latency(model, cfg: Config, n_warm=10, n_iter=50):
    model.eval()
    x = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=cfg.device)
    for _ in range(n_warm): model(x)
    if cfg.device == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter): model(x)
    if cfg.device == "cuda": torch.cuda.synchronize()
    ms = (time.time() - t0) / n_iter * 1000
    return {"latency_ms": ms, "fps": 1000.0 / ms}

def efficiency_report(model, cfg: Config, name="model"):
    rep = {"name": name, "params_M": count_params(model) / 1e6, "gflops": measure_flops(model, cfg)}
    rep.update(measure_latency(model, cfg))
    return rep


# ------------------------------------------------------------------ #
# 5. Training Loops (Resumable, Crash-Proof, OOM-safe)
# ------------------------------------------------------------------ #
def cosine_warmup(optimizer, base_lr, epoch, warmup, total):
    if epoch < warmup: lr = base_lr * (epoch + 1) / max(1, warmup)
    else:
        prog = (epoch - warmup) / max(1, total - warmup)
        lr = 0.5 * base_lr * (1 + math.cos(math.pi * prog))
    for g in optimizer.param_groups: g["lr"] = lr
    return lr


@torch.no_grad()
def evaluate(model, loader, cfg: Config, class_names, test_counts):
    model.eval()
    ious, dices, labels, y_true, y_pred = [], [], [], [], []
    for b in loader:
        img = b["image"].to(cfg.device); msk = b["mask"].to(cfg.device); lab = b["label"]
        out = model(img)
        i, d = seg_scores(out["seg"], msk)
        ious.append(i); dices.append(d); labels.append(lab.numpy())
        y_true.append(lab.numpy()); y_pred.append(out["cls"].argmax(1).cpu().numpy())
    return {
        "segmentation": summarize_seg(np.concatenate(ious), np.concatenate(dices), np.concatenate(labels), class_names, test_counts),
        "classification": summarize_cls(np.concatenate(y_true), np.concatenate(y_pred), class_names)
    }

@torch.no_grad()
def evaluate_cls(model, loader, cfg, class_names):
    model.eval()
    yt, yp = [], []
    for b in loader:
        img = b["image"].to(cfg.device)
        out = model(img, compute_seg=False)
        yt.append(b["label"].numpy()); yp.append(out["cls"].argmax(1).cpu().numpy())
    return summarize_cls(np.concatenate(yt), np.concatenate(yp), class_names)


def _train_multitask(model, teacher, train_loader, val_loader, cfg, class_names, test_counts, epochs, tag):
    from tqdm.auto import tqdm
    model.to(cfg.device)
    if teacher is not None:
        teacher.to(cfg.device).eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        
    crit = DistillLoss(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and cfg.device == "cuda"))
    
    os.makedirs(cfg.out_dir, exist_ok=True)
    hist_path = os.path.join(cfg.out_dir, f"{tag}_history.json")
    best_weights_path = os.path.join(cfg.out_dir, f"{tag}_best.pt")
    
    existing_hist, start_ep, best_score = [], 0, -1.0
    if os.path.exists(hist_path):
        try:
            with open(hist_path, "r") as f: existing_hist = json.load(f)
            start_ep = len(existing_hist)
            if existing_hist: best_score = max([h.get("val_wmIoU", 0.0) for h in existing_hist])
        except Exception: pass
        
    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=cfg.device))
        print(f"[{tag}] Resuming from epoch {start_ep} with best wmIoU {best_score:.4f}")

    best, best_state, wait = best_score, None, 0

    for ep in range(start_ep, epochs):
        model.train()
        lr = cosine_warmup(opt, cfg.lr, ep, cfg.warmup_epochs, epochs)
        run = {}
        pbar = tqdm(train_loader, desc=f"[{tag}] ep {ep+1}/{epochs}", leave=False)
        for b in pbar:
            img = b["image"].to(cfg.device, non_blocking=True)
            msk = b["mask"].to(cfg.device, non_blocking=True)
            lab = b["label"].to(cfg.device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=(cfg.amp and cfg.device == "cuda")):
                s_out = model(img)
                t_out = None
                if teacher is not None:
                    with torch.no_grad(): t_out = teacher(img)
                loss, parts = crit(s_out, t_out, msk, lab)
                
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            
            for k, v in parts.items(): run[k] = run.get(k, 0.0) + v
            pbar.set_postfix(loss=f"{parts['total']:.3f}")
            
        run = {k: v / len(train_loader) for k, v in run.items()}
        metrics = evaluate(model, val_loader, cfg, class_names, test_counts)
        score = metrics["segmentation"]["weighted_mIoU"]
        
        epoch_log = {"epoch": ep + 1, "lr": lr, **run, "val_wmIoU": score, "val_acc": metrics["classification"]["accuracy"]}
        existing_hist.append(epoch_log)
        print(f"[{tag}] ep {ep+1}: loss={run.get('total',0):.3f} val_wmIoU={score:.4f} val_acc={metrics['classification']['accuracy']:.4f}")

        if score > best:
            best, best_state, wait = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
            torch.save(best_state, best_weights_path)
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"[{tag}] early stop @ ep {ep+1}")
                break
                
        with open(hist_path, "w") as f: json.dump(existing_hist, f, indent=2)
        gc.collect(); torch.cuda.empty_cache()

    if best_state is not None: model.load_state_dict(best_state)
    return model, existing_hist


def _train_cls_bulletproof(model, teacher, train_loader, val_loader, cfg, class_names, epochs, tag):
    from tqdm.auto import tqdm
    model.to(cfg.device)
    if teacher is not None:
        teacher.to(cfg.device).eval()
        teacher.float() # Strict FP32 for the teacher to prevent AMP overflow
        for p in teacher.parameters(): p.requires_grad_(False)
            
    crit = ClsDistillLossStable(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and cfg.device == "cuda"))
    
    os.makedirs(cfg.out_dir, exist_ok=True)
    hist_path = os.path.join(cfg.out_dir, f"{tag}_history.json")
    best_weights_path = os.path.join(cfg.out_dir, f"{tag}_best.pt")
    
    existing_hist, start_ep, best_acc = [], 0, -1.0
    if os.path.exists(hist_path):
        try:
            with open(hist_path, "r") as f: existing_hist = json.load(f)
            start_ep = len(existing_hist)
            if existing_hist: best_acc = max([h.get("val_acc", 0.0) for h in existing_hist])
        except Exception: pass
        
    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=cfg.device))
        print(f"[{tag}] Resuming from epoch {start_ep} with best acc {best_acc:.4f}")

    best, best_state, wait = best_acc, None, 0

    for ep in range(start_ep, epochs):
        model.train()
        lr = cosine_warmup(opt, cfg.lr, ep, cfg.warmup_epochs, epochs)
        run = {}
        pbar = tqdm(train_loader, desc=f"[{tag}] ep {ep+1}/{epochs}", leave=False)
        for b in pbar:
            img = b["image"].to(cfg.device, non_blocking=True)
            lab = b["label"].to(cfg.device, non_blocking=True)
            
            t_out = None
            if teacher is not None:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                    t_out = teacher(img.float(), compute_seg=False)
            
            with torch.cuda.amp.autocast(enabled=(cfg.amp and cfg.device == "cuda")):
                s_out = model(img, compute_seg=False)
                loss, parts = crit(s_out, t_out, lab)
                
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            
            for k, v in parts.items(): run[k] = run.get(k, 0.0) + v
            pbar.set_postfix(loss=f"{parts['total']:.3f}")
            
        run = {k: v / len(train_loader) for k, v in run.items()}
        m = evaluate_cls(model, val_loader, cfg, class_names)
        acc = m["accuracy"]
        
        epoch_log = {"epoch": ep + 1, "lr": lr, **run, "val_acc": acc, "val_wF1": m["weighted"]["f1"]}
        existing_hist.append(epoch_log)
        print(f"[{tag}] ep {ep+1}: loss={run.get('total',0):.3f} val_acc={acc:.4f} val_wF1={m['weighted']['f1']:.4f}")
        
        if acc > best:
            best, best_state, wait = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
            torch.save(best_state, best_weights_path)
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"[{tag}] early stop @ ep {ep+1}")
                break
                
        with open(hist_path, "w") as f: json.dump(existing_hist, f, indent=2)
        gc.collect(); torch.cuda.empty_cache()

    if best_state is not None: model.load_state_dict(best_state)
    return model, existing_hist


# ------------------------------------------------------------------ #
# 6. Visualisation
# ------------------------------------------------------------------ #
class GradCAM:
    def __init__(self, model: MultiTaskNet):
        self.model = model
        self.acts = None; self.grads = None
        self._h1 = model.encoder.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        f = out[-1]
        if f.requires_grad:
            f.retain_grad()
            self.acts = f; self._target = f

    def __call__(self, x, class_idx=None):
        self.model.eval()
        out = self.model(x)
        logits = out["cls"]
        if class_idx is None: class_idx = logits.argmax(1)
        self.model.zero_grad()
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        score.backward(retain_graph=False)
        grads = self._target.grad
        acts = self._target.detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        cam = (cam - cam.amin(dim=(1, 2), keepdim=True)) / (cam.amax(dim=(1, 2), keepdim=True) - cam.amin(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach().cpu().numpy(), class_idx.cpu().numpy()

    def close(self): self._h1.remove()

def _denorm(img_t):
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    return np.clip(img_t.cpu().numpy().transpose(1, 2, 0) * std + mean, 0, 1)

def plot_history(hist, path):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h.get("total", np.nan) for h in hist], label="train loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Training loss")
    ax[1].plot(ep, [h["val_wmIoU"] for h in hist], label="val wmIoU")
    ax[1].plot(ep, [h["val_acc"] for h in hist], label="val acc")
    ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_title("Validation metrics")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

def plot_cls_history(hist, path):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    loss_vals = [h.get("total", h.get("cls", np.nan)) for h in hist]
    ax[0].plot(ep, loss_vals, label="train loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Training Loss")
    ax[1].plot(ep, [h["val_acc"] for h in hist], label="val accuracy")
    ax[1].plot(ep, [h["val_wF1"] for h in hist], label="val weighted F1")
    ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_title("Validation Metrics")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

def plot_confusion(cm, class_names, path):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cm = np.asarray(cm, float)
    cmn = cm / (cm.sum(1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right"); ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{int(cm[i,j])}", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

def save_qualitative(model, loader, cfg, class_names, path, n=6):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.eval(); cam = GradCAM(model)
    batch = next(iter(loader))
    img = batch["image"][:n].to(cfg.device); msk = batch["mask"][:n]
    with torch.no_grad(): out = model(img)
    pred = (torch.sigmoid(out["seg"]) > 0.5).float().cpu()
    heat, pcls = cam(img); cam.close()
    fig, ax = plt.subplots(n, 4, figsize=(11, 2.6 * n))
    if n == 1: ax = ax[None, :]
    for i in range(n):
        base = _denorm(img[i])
        ax[i, 0].imshow(base); ax[i, 0].set_title("image"); ax[i, 0].axis("off")
        ax[i, 1].imshow(msk[i, 0], cmap="gray"); ax[i, 1].set_title("GT mask"); ax[i, 1].axis("off")
        ax[i, 2].imshow(pred[i, 0], cmap="gray"); ax[i, 2].set_title("pred mask"); ax[i, 2].axis("off")
        ax[i, 3].imshow(base); ax[i, 3].imshow(heat[i], cmap="jet", alpha=0.45)
        ax[i, 3].set_title(f"Grad-CAM: {class_names[pcls[i]]}"); ax[i, 3].axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ #
# 7. Execution Orchestrators
# ------------------------------------------------------------------ #
def build_results_table(teacher_metrics, student_metrics, teacher_eff, student_eff, path):
    import pandas as pd
    rows = []
    for tag, m, e in [("Teacher", teacher_metrics, teacher_eff), ("Student", student_metrics, student_eff)]:
        s, c = m["segmentation"], m["classification"]
        rows.append({
            "Model": tag, "Params (M)": round(e["params_M"], 2),
            "GFLOPs": (round(e["gflops"], 2) if e["gflops"] else None),
            "Latency (ms)": round(e["latency_ms"], 2),
            "IoU glioma": round(s["per_class_iou"]["glioma"] * 100, 2),
            "IoU mening.": round(s["per_class_iou"]["meningioma"] * 100, 2),
            "IoU pituit.": round(s["per_class_iou"]["pituitary"] * 100, 2),
            "wmIoU": round(s["weighted_mIoU"] * 100, 2),
            "Dice": round(s["mean_dice"] * 100, 2),
            "Cls Acc": round(c["accuracy"] * 100, 2),
            "Cls wF1": round(c["weighted"]["f1"] * 100, 2),
        })
    df = pd.DataFrame(rows); df.to_csv(path, index=False)
    return df


def build_cls_table(teacher_m, student_m, teacher_eff, student_eff, path):
    import pandas as pd
    rows = []
    for tag, m, e in [("Teacher", teacher_m, teacher_eff), ("Student (ours)", student_m, student_eff)]:
        rows.append({
            "Model": tag, "Params (M)": round(e["params_M"], 2),
            "GFLOPs": (round(e["gflops"], 2) if e["gflops"] else None),
            "Latency (ms)": round(e["latency_ms"], 2),
            "Accuracy": round(m["accuracy"] * 100, 2),
            "Weighted F1": round(m["weighted"]["f1"] * 100, 2),
        })
    df = pd.DataFrame(rows); df.to_csv(path, index=False)
    return df


def main(cfg: Config, do_ablation: bool = True):
    """Full Multi-Task Track"""
    set_seed(cfg.seed); cfg.dump()
    K = len(cfg.class_names)
    train_loader, val_loader, test_loader, test_counts = make_loaders(cfg)

    print("\n==================== TEACHER ====================")
    teacher = build_model("teacher", cfg, K)
    teacher, t_hist = _train_multitask(teacher, None, train_loader, val_loader, cfg, cfg.class_names, test_counts, cfg.epochs_teacher, "teacher")
    t_metrics = evaluate(teacher, test_loader, cfg, cfg.class_names, test_counts)
    t_eff = efficiency_report(teacher, cfg, cfg.teacher_encoder)
    plot_history(t_hist, os.path.join(cfg.out_dir, "teacher_curves.png"))

    print("\n==================== STUDENT (SPKD) ====================")
    student = build_model("student", cfg, K)
    student, s_hist = _train_multitask(student, teacher, train_loader, val_loader, cfg, cfg.class_names, test_counts, cfg.epochs_student, "student_full")
    s_metrics = evaluate(student, test_loader, cfg, cfg.class_names, test_counts)
    s_eff = efficiency_report(student, cfg, cfg.student_encoder)
    
    plot_history(s_hist, os.path.join(cfg.out_dir, "student_curves.png"))
    plot_confusion(s_metrics["classification"]["confusion_matrix"], cfg.class_names, os.path.join(cfg.out_dir, "student_confusion.png"))
    save_qualitative(student, test_loader, cfg, cfg.class_names, os.path.join(cfg.out_dir, "student_qualitative.png"))

    df = build_results_table(t_metrics, s_metrics, t_eff, s_eff, os.path.join(cfg.out_dir, "results_main.csv"))
    print("\n==================== MAIN RESULTS ====================")
    print(df.to_string(index=False))

    abl = None
    if do_ablation:
        print("\n==================== ABLATION ====================")
        settings = {
            "S0_no_KD": dict(kd_seg=0.0, kd_cls=0.0, spkd_w=0.0),
            "S1_logitKD": dict(kd_seg=cfg.kd_seg, kd_cls=cfg.kd_cls, spkd_w=0.0),
            "S2_SPKD_only": dict(kd_seg=0.0, kd_cls=0.0, spkd_w=cfg.spkd_w),
            "S3_full": dict(kd_seg=cfg.kd_seg, kd_cls=cfg.kd_cls, spkd_w=cfg.spkd_w),
        }
        abl = []
        for name, ov in settings.items():
            acfg = copy.deepcopy(cfg)
            for k, v in ov.items(): setattr(acfg, k, v)
            set_seed(cfg.seed)
            st = build_model("student", cfg, K)
            tea = teacher if (ov["kd_seg"] or ov["kd_cls"] or ov["spkd_w"]) else None
            st, _ = _train_multitask(st, tea, train_loader, val_loader, acfg, cfg.class_names, test_counts, cfg.epochs_student, f"abl_{name}")
            m = evaluate(st, test_loader, acfg, cfg.class_names, test_counts)
            abl.append({
                "setting": name, "weighted_mIoU": m["segmentation"]["weighted_mIoU"],
                "mean_dice": m["segmentation"]["mean_dice"], "cls_acc": m["classification"]["accuracy"]
            })
        import pandas as pd
        pd.DataFrame(abl).to_csv(os.path.join(cfg.out_dir, "ablation.csv"), index=False)

    with open(os.path.join(cfg.out_dir, "final_metrics.json"), "w") as f:
        json.dump({"teacher": {**t_metrics, "efficiency": t_eff}, "student": {**s_metrics, "efficiency": s_eff}, "ablation": abl}, f, indent=2)
    return {"teacher": teacher, "student": student, "table": df, "ablation": abl}


def main_classification(cfg: Config, do_ablation: bool = False):
    """Standalone 4-Class Classification Track (FP16 Overflow Resilient)"""
    # Enforce pure FP32 for the distillation step to prevent KL divergence overflow
    cfg.amp = False
    set_seed(cfg.seed); cfg.dump()
    K = len(cfg.cls_class_names)
    train_loader, val_loader, test_loader = make_cls_loaders(cfg)

    print("\n============ CLS TEACHER (4-class) ============")
    teacher = build_model("teacher", cfg, K)
    teacher, th = _train_cls_bulletproof(teacher, None, train_loader, val_loader, cfg, cfg.cls_class_names, cfg.epochs_teacher, "cls_teacher")
    t_m = evaluate_cls(teacher, test_loader, cfg, cfg.cls_class_names)
    t_eff = efficiency_report(teacher, cfg, cfg.teacher_encoder)

    print("\n============ CLS STUDENT (SPKD, 4-class) ============")
    student = build_model("student", cfg, K)
    student, sh = _train_cls_bulletproof(student, teacher, train_loader, val_loader, cfg, cfg.cls_class_names, cfg.epochs_student, "cls_student")
    s_m = evaluate_cls(student, test_loader, cfg, cfg.cls_class_names)
    s_eff = efficiency_report(student, cfg, cfg.student_encoder)
    
    plot_cls_history(sh, os.path.join(cfg.out_dir, "cls_student_curves.png"))
    plot_confusion(s_m["confusion_matrix"], cfg.cls_class_names, os.path.join(cfg.out_dir, "cls_student_confusion.png"))

    df = build_cls_table(t_m, s_m, t_eff, s_eff, os.path.join(cfg.out_dir, "results_classification.csv"))
    print("\n============ 4-CLASS RESULTS ============")
    print(df.to_string(index=False))

    abl = None
    if do_ablation:
        print("\n============ RUNNING ABLATIONS ============")
        import pandas as pd
        settings = {
            "S0_no_KD": dict(kd_cls=0.0, spkd_w=0.0),
            "S1_logitKD": dict(kd_cls=cfg.kd_cls, spkd_w=0.0),
            "S2_SPKD_only": dict(kd_cls=0.0, spkd_w=cfg.spkd_w),
            "S3_full": dict(kd_cls=cfg.kd_cls, spkd_w=cfg.spkd_w)
        }
        abl = []
        for name, ov in settings.items():
            acfg = copy.deepcopy(cfg)
            for k, v in ov.items(): setattr(acfg, k, v)
            set_seed(cfg.seed)
            st = build_model("student", cfg, K)
            tea = teacher if (ov["kd_cls"] or ov["spkd_w"]) else None
            st, _ = _train_cls_bulletproof(st, tea, train_loader, val_loader, acfg, cfg.cls_class_names, cfg.epochs_student, f"cls_abl_{name}")
            m = evaluate_cls(st, test_loader, acfg, cfg.cls_class_names)
            abl.append({"setting": name, "accuracy": m["accuracy"], "weighted_f1": m["weighted"]["f1"]})
        pd.DataFrame(abl).to_csv(os.path.join(cfg.out_dir, "cls_ablation.csv"), index=False)

    with open(os.path.join(cfg.out_dir, "final_metrics_classification.json"), "w") as f:
        json.dump({"teacher": {**t_m, "efficiency": t_eff}, "student": {**s_m, "efficiency": s_eff}, "ablation": abl}, f, indent=2)
    return {"teacher": teacher, "student": student, "table": df, "ablation": abl}
