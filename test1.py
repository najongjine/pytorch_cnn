# ============================================================
# Dogs vs Cats - PyTorch CNN (초간단 데모 버전)
# - 큰 데이터 X, 인터넷에서 고양이/강아지 샘플 이미지만 소량 다운로드
# - ImageFolder 구조 자동 생성: dvc_data/{train,val}/{cat,dog}
# - EDU_MODE=True면 중간 텐서 shape 로그 + 간단한 feature map 시각화(선택)
# ============================================================

import os, io, time, random
from pathlib import Path
from typing import List

import requests
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F

# -----------------------
# 0) 설정
# -----------------------
SEED = 42
random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

BASE_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
DATA_ROOT = BASE_DIR / "dvc_data"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR   = DATA_ROOT / "val"
SMALL_DEMO = False        # 소량 샘플만 받을지 여부
N_PER_CLASS = 20         # 클래스당 다운로드 장수(훈련용 기준). 너무 크면 느려짐
VAL_RATIO = 0.1          # 9:1 분할
EDU_MODE = True          # 중간 로그/시각화 켜기
IMG_SIZE = 128
BATCH = 64
EPOCHS = 3               # 맛보기 학습 에폭

# -----------------------
# 1) 소량 샘플 자동 다운로드 (고양이/강아지)
# -----------------------

def _ensure_dirs():
    for p in [TRAIN_DIR/"cat", TRAIN_DIR/"dog", VAL_DIR/"cat", VAL_DIR/"dog"]:
        p.mkdir(parents=True, exist_ok=True)

def _save_from_urls(urls: List[str], out_dir: Path, prefix: str) -> int:
    ok = 0
    for i, url in enumerate(urls):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.save(out_dir / f"{prefix}_{i:03d}.jpg", format="JPEG", quality=90)
            ok += 1
        except Exception:
            pass
        time.sleep(0.05)   # 연속요청 살짝 완화
    return ok

def _fetch_urls_dog(n: int) -> List[str]:
    try:
        r = requests.get(f"https://dog.ceo/api/breeds/image/random/{n}", timeout=10)
        r.raise_for_status()
        return list(r.json().get("message", []) or [])
    except Exception:
        return []

def _fetch_urls_cat(n: int) -> List[str]:
    # cataas 랜덤 고양이
    return [f"https://cataas.com/cat?width=512&height=512&rand={i}" for i in range(n)]


def tiny_fetch_if_missing(n_per_cls=N_PER_CLASS, val_ratio=VAL_RATIO):
    has_train = TRAIN_DIR.exists() and any((TRAIN_DIR/"cat").glob("*.jpg")) and any((TRAIN_DIR/"dog").glob("*.jpg"))
    if has_train:
        return
    print(f"[tiny] 로컬 데이터 없음 → 소량 샘플 다운로드(cat/dog 각 {n_per_cls}장)…")
    _ensure_dirs()
    cat_ok = _save_from_urls(_fetch_urls_cat(n_per_cls), TRAIN_DIR/"cat", "cat")
    dog_ok = _save_from_urls(_fetch_urls_dog(n_per_cls), TRAIN_DIR/"dog", "dog")
    print(f"[tiny] train 준비: cat={cat_ok}, dog={dog_ok}")

    # 간단 분할(9:1)
    cats = sorted((TRAIN_DIR/"cat").glob("*.jpg"))
    dogs = sorted((TRAIN_DIR/"dog").glob("*.jpg"))
    random.shuffle(cats); random.shuffle(dogs)
    nvc = max(1, int(len(cats)*val_ratio)) if len(cats)>0 else 0
    nvd = max(1, int(len(dogs)*val_ratio)) if len(dogs)>0 else 0
    for p in cats[:nvc]: p.replace(VAL_DIR/"cat"/p.name)
    for p in dogs[:nvd]: p.replace(VAL_DIR/"dog"/p.name)
    print(f"[tiny] 분할 완료: val(cat)={nvc}, val(dog)={nvd}")

# -----------------------
# 2) 데이터 준비
# -----------------------
if SMALL_DEMO:
    tiny_fetch_if_missing()
else:
    # 수동으로 dvc_data/train/{cat,dog}에 이미지 넣어도 됨
    pass

# 변환(전처리)
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

# 데이터셋 빌드
has_train = TRAIN_DIR.exists() and any((TRAIN_DIR/"cat").glob("*.jpg")) and any((TRAIN_DIR/"dog").glob("*.jpg"))
if not has_train:
    raise FileNotFoundError(
        f"데이터가 없습니다. 다음 구조로 몇 장만 넣어도 됩니다:\n"
        f"{TRAIN_DIR}/cat/*.jpg, {TRAIN_DIR}/dog/*.jpg\n"
        f"옵션) {VAL_DIR}/cat/*.jpg, {VAL_DIR}/dog/*.jpg"
    )

full_train = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
# val 폴더가 충분하면 그대로 사용, 아니면 9:1 분할
has_val = VAL_DIR.exists() and any((VAL_DIR/"cat").glob("*.jpg")) and any((VAL_DIR/"dog").glob("*.jpg"))
if has_val:
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tf)
    train_ds = full_train
else:
    n_total = len(full_train)
    n_val = max(1, int(n_total * VAL_RATIO))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    val_ds.dataset.transform = val_tf

classes = train_ds.dataset.classes if hasattr(train_ds, "dataset") else train_ds.classes
print("classes:", classes)

# DataLoader (Windows 안정화를 위해 num_workers=0 권장)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False)

# -----------------------
# 3) 간단 CNN
# -----------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------
# 4) (옵션) 교육용 시각화/로그
# -----------------------
import matplotlib.pyplot as plt

MAX_CH = 32

def _minmax(t):
    t = t.detach().cpu()
    return float(t.min()), float(t.max())

def show_feature_map(chw, title="Feature Map", max_channels=MAX_CH, ncols=8):
    t = chw.detach().cpu()
    C, H, W = t.shape
    C = min(C, max_channels)
    nrows = (C + ncols - 1) // ncols
    plt.figure(figsize=(ncols*2, nrows*2))
    for i in range(C):
        ch = t[i]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        ax = plt.subplot(nrows, ncols, i+1)
        ax.imshow(ch.numpy(), cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def visualize_conv_kernels(layer_idx=0):
    # layer_idx: 0=Conv1, 4=Conv2, 8=Conv3 (현재 모델 features 인덱스)
    conv = model.features[layer_idx]
    W = conv.weight.detach().cpu()            # (out, in, kh, kw)
    Wm = W.mean(dim=1)                        # RGB 평균 → 1채널로 간단 확인
    C, H, Wk = Wm.shape
    ncols = 8; nrows = (C + ncols - 1) // ncols
    import matplotlib.pyplot as plt
    plt.figure(figsize=(ncols*2, nrows*2))
    for i in range(C):
        k = Wm[i]
        k = (k - k.min()) / (k.max() - k.min() + 1e-8)
        ax = plt.subplot(nrows, ncols, i+1)
        ax.imshow(k.numpy(), cmap="gray"); ax.axis("off")
    plt.suptitle(f"Conv kernels @ layer_idx={layer_idx}")
    plt.tight_layout(); plt.show()

def _denorm_img(img3chw):
    mean = torch.tensor([0.5,0.5,0.5], device=img3chw.device).view(3,1,1)
    std  = torch.tensor([0.5,0.5,0.5], device=img3chw.device).view(3,1,1)
    return (img3chw*std + mean).clamp(0,1).permute(1,2,0).cpu().numpy()

def overlay_topk_activations(xb_cpu, fmap, k=6, title=""):
    """xb_cpu: (1,3,H,W) 원본, fmap: (1,C,h,w) 해당 레이어 출력"""
    x_np = _denorm_img(xb_cpu[0])
    A = fmap.detach().cpu()[0]               # (C,h,w)
    # 채널별 평균 활성도가 큰 k개 선택
    scores = A.view(A.shape[0], -1).mean(dim=1)
    topk = torch.topk(scores, k=min(k, A.shape[0]))[1].tolist()

    H, W = xb_cpu.shape[2], xb_cpu.shape[3]
    ncols = 3; nrows = (len(topk) + ncols - 1)//ncols
    plt.figure(figsize=(ncols*4, nrows*4))
    for i, ch in enumerate(topk):
        act = A[ch:ch+1]                     # (1,h,w)
        act = act - act.min()
        act = act / (act.max() + 1e-8)
        act_up = F.interpolate(act.unsqueeze(0), size=(H,W), mode="bilinear", align_corners=False)[0,0].numpy()

        # overlay
        heat = plt.cm.jet(act_up)[..., :3]   # (H,W,3)
        mix = (0.5*x_np + 0.5*heat)
        ax = plt.subplot(nrows, ncols, i+1)
        ax.imshow(mix); ax.set_title(f"{title} ch={ch}"); ax.axis("off")
    plt.tight_layout(); plt.show()

def forward_to_idx(x, last_idx):
    y = x
    for i in range(last_idx+1):
        y = model.features[i](y)
    return y

# Grad-CAM for last conv block (relu3 @ features[10])
def grad_cam(xb_cpu, class_idx=None):
    model.eval()
    torch.set_grad_enabled(True)                 # 혹시 모를 no_grad 상태 해제
    for p in model.parameters():
        p.requires_grad_(True)                   # 파라미터 grad 허용

    feats = None
    grads = None

    def fwd_hook(m, i, o):
        nonlocal feats; feats = o                # (B,C,h,w)
    def bwd_hook(g):
        nonlocal grads; grads = g                # (B,C,h,w)

    # 마지막 conv 뒤 ReLU 출력에 훅(당신 모델에선 features[10] = ReLU3)
    h1 = model.features[10].register_forward_hook(fwd_hook)
    h2 = model.features[10].register_full_backward_hook(lambda m, gin, gout: bwd_hook(gout[0]))

    # ⚠️ leaf 텐서로 만들어서 requires_grad=True
    x = xb_cpu.to(DEVICE)
    x = x.clone().detach().requires_grad_(True)

    logits = model(x)                            # (B,num_classes)
    if class_idx is None:
        class_idx = int(logits.argmax(dim=1)[0])
    score = logits[0, class_idx]                 # 그래프에 연결된 0-dim tensor

    model.zero_grad()
    score.backward(retain_graph=False)           # ← 이제 grad_fn 존재

    h1.remove(); h2.remove()

    # 채널 가중치: d(score)/d(feats) 의 GAP
    w = grads[0].detach().cpu().mean(dim=(1,2))  # (C,)
    cam = (w.view(-1,1,1) * feats[0].detach().cpu()).sum(dim=0)
    cam = torch.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    H, W = xb_cpu.shape[2], xb_cpu.shape[3]
    cam_up = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                           size=(H, W), mode="bilinear",
                           align_corners=False)[0,0].numpy()

    img = _denorm_img(xb_cpu[0])
    heat = plt.cm.jet(cam_up)[..., :3]
    mix = 0.4*img + 0.6*heat
    plt.figure(figsize=(4,4))
    plt.imshow(mix); plt.title(f"Grad-CAM (class={classes[class_idx]})")
    plt.axis("off"); plt.show()



@torch.no_grad()
def explain_and_visualize_one(xb_cpu):
    model.eval()
    with torch.no_grad():
        x = xb_cpu.to(DEVICE)

        # Block1
        conv1 = model.features[0](x); bn1 = model.features[1](conv1); relu1 = model.features[2](bn1); pool1 = model.features[3](relu1)
        print("[Block1] Conv→BN→ReLU→Pool")
        print("  Conv1:", conv1.shape, "minmax=", _minmax(conv1[0]))
        print("   BN1 :", bn1.shape,   "minmax=", _minmax(bn1[0]))
        print("  ReLU1:", relu1.shape, "minmax=", _minmax(relu1[0]))
        print("  Pool1:", pool1.shape, "minmax=", _minmax(pool1[0]), "← 해상도 절반으로, 노이즈 감소/요약")

        # Block2
        conv2 = model.features[4](pool1); bn2 = model.features[5](conv2); relu2 = model.features[6](bn2); pool2 = model.features[7](relu2)
        print("[Block2] Conv→BN→ReLU→Pool (더 복잡한 패턴)")
        print("  Conv2:", conv2.shape, "minmax=", _minmax(conv2[0]))
        print("   BN2 :", bn2.shape,   "minmax=", _minmax(bn2[0]))
        print("  ReLU2:", relu2.shape, "minmax=", _minmax(relu2[0]))
        print("  Pool2:", pool2.shape, "minmax=", _minmax(pool2[0]))

        # Block3
        conv3 = model.features[8](pool2); bn3 = model.features[9](conv3); relu3 = model.features[10](bn3); pool3 = model.features[11](relu3)
        print("[Block3] Conv→BN→ReLU→Pool (귀/코/수염 같은 조합 특징)")
        print("  Conv3:", conv3.shape, "minmax=", _minmax(conv3[0]))
        print("   BN3 :", bn3.shape,   "minmax=", _minmax(bn3[0]))
        print("  ReLU3:", relu3.shape, "minmax=", _minmax(relu3[0]))
        print("  Pool3:", pool3.shape, "minmax=", _minmax(pool3[0]), "→ 마지막 요약본")

        # 분류기 앞/뒤
        flat = model.classifier[0](pool3); h1 = model.classifier[1](flat); h1a = model.classifier[2](h1); out = model.classifier[4](h1a)
        print("[Classifier] Flatten→Linear(128)→ReLU→Dropout→Linear(2)")
        print("  Flatten:", flat.shape)
        print("  Hidden :", h1.shape, "ReLU 후:", h1a.shape)
        print("  Logits :", out.shape, "예: ", out[0].detach().cpu().tolist(), " (softmax 전 점수)")

        # === 시각화(배치 첫 이미지 기준) ===
        img0 = xb_cpu[0]  # (3, H, W)

        # ✅ 정규화 해제: x = x*std + mean  (브로드캐스팅 되도록 (3,1,1) 모양)
        mean = torch.tensor([0.5, 0.5, 0.5], device=img0.device).view(3,1,1)
        std  = torch.tensor([0.5, 0.5, 0.5], device=img0.device).view(3,1,1)
        inv = (img0 * std + mean).clamp(0, 1).permute(1,2,0).cpu().numpy()

        import matplotlib.pyplot as plt
        plt.figure(figsize=(3,3)); plt.imshow(inv); plt.title("input image(recover)"); plt.axis("off"); plt.show()

        # 1층 커널 모양
        visualize_conv_kernels(0)

        # 레이어별 top-k 활성화 오버레이
        xb_vis, _ = next(iter(val_loader))
        xb1 = xb_vis[:1]
        conv1 = forward_to_idx(xb1.to(DEVICE), 0)
        conv2 = forward_to_idx(xb1.to(DEVICE), 4)
        conv3 = forward_to_idx(xb1.to(DEVICE), 8)
        overlay_topk_activations(xb1, conv1, k=6, title="Conv1")
        overlay_topk_activations(xb1, conv2, k=6, title="Conv2")
        overlay_topk_activations(xb1, conv3, k=6, title="Conv3")

        # 최종 분류 근거 위치
        grad_cam(xb1)



        show_feature_map(conv1[0].detach().cpu(), "Conv1 Output: Edges / Dots / Simple Patterns")
        show_feature_map(pool1[0].detach().cpu(), "Pool1 Output: Reduced Resolution, Summarized")
        show_feature_map(conv2[0].detach().cpu(), "Conv2 Output: More Complex Patterns")
        show_feature_map(conv3[0].detach().cpu(), "Conv3 Output: Combined Features (e.g., ears, nose, whiskers)")

        h1_img = h1.squeeze().cpu().numpy().reshape(8, 16)  # or (16,8)
        plt.figure(figsize=(6, 3))
        plt.imshow(h1_img, cmap='viridis')
        plt.title("Linear(64*16*16 → 128) Output as Image")
        plt.colorbar()
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        




# -----------------------
# 5) 학습 루프
# -----------------------

def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

for epoch in range(1, EPOCHS+1):
    # Train
    model.train()
    tot_loss = tot_acc = n = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward(); optimizer.step()
        bs = xb.size(0)
        tot_loss += loss.item()*bs
        tot_acc  += accuracy(logits, yb)*bs
        n += bs
    train_loss, train_acc = tot_loss/n, tot_acc/n

    # Validate
    model.eval()
    v_loss = v_acc = vn = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            bs = xb.size(0)
            v_loss += loss.item()*bs
            v_acc  += accuracy(logits, yb)*bs
            vn += bs
    print(f"[{epoch}/{EPOCHS}] train_loss={train_loss:.4f} acc={train_acc:.3f} | val_loss={v_loss/vn:.4f} acc={v_acc/vn:.3f}")

    if EDU_MODE and (epoch == 1 or epoch == EPOCHS):
        xb_vis, _ = next(iter(val_loader))
        explain_and_visualize_one(xb_vis[:1].cpu())

# -----------------------
# 6) 예측 함수
# -----------------------
softmax = nn.Softmax(dim=1)

def predict_image(img_path: str):
    img = Image.open(img_path).convert("RGB")
    x = val_tf(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = softmax(logits).squeeze().cpu().tolist()
        idx = int(torch.argmax(logits, dim=1))
        return (classes[idx], probs)

print("Done. 예: predict_image('dvc_data/val/cat/cat_000.jpg')")
