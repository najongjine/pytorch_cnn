# ============================================================
# Dogs vs Cats - PyTorch CNN (교육용)
# - 데이터가 이미 있다면 ImageFolder 구조로 바로 사용:
#     data/
#       train/
#         cat/  *.jpg
#         dog/  *.jpg
#       val/    (없으면 자동 분할)
#         cat/  *.jpg
#         dog/  *.jpg
# - EDU_MODE=True 이면 각 블록별 출력 shape + 값범위 로그 + feature map 시각화
# ============================================================

from torchvision.datasets import OxfordIIITPet
import os, random, math, glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from shutil import copy2
from PIL import Image

# -----------------------
# 0) 설정
# -----------------------
SEED = 42
random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

# === 경로 설정 ===
# 1) 직접 가진 데이터 사용 (권장): 아래 DATA_ROOT를 본인 폴더로 바꾸면 됨
#    구조: DATA_ROOT/{train,val}/{cat,dog}/*.jpg
#DATA_ROOT = "/content/dvc_data"
BASE_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
DATA_ROOT = BASE_DIR / "dvc_data"   # 현재 경로/dvc_data

# 2) Kaggle Dogs vs Cats 원본을 쓰고 싶다면:
#    RAW_DIR = "/kaggle/input/dogs-vs-cats/train"
#    폴더 자동분류 유틸을 하나 더 붙일 수 있음(여기선 생략)
os.makedirs(DATA_ROOT, exist_ok=True)
train_dir = Path(DATA_ROOT) / "train"
val_dir   = Path(DATA_ROOT) / "val"


def prepare_imagefolder(DATA_ROOT: Path, train_dir: Path, val_dir: Path):
    """
    다음과 같은 흔한 케이스를 자동 정리:
    1) DATA_ROOT 아래에 jpg가 'cat'/'dog'가 섞여만 있고(train/cat 폴더 없음)
    2) DATA_ROOT/train 아래에 jpg가 섞여만 있고 (cat/dog 폴더 없음)
    파일명에 'cat' 또는 'dog'가 포함되면 해당 클래스로 분류해 복사.
    (원본은 그대로 두고 복사: copy2)
    """
    def ensure_cls_dirs(root: Path):
        (root/"cat").mkdir(parents=True, exist_ok=True)
        (root/"dog").mkdir(parents=True, exist_ok=True)

    def move_by_name(src_glob_pattern, dst_root: Path):
        ensure_cls_dirs(dst_root)
        moved = 0
        for p in Path().glob(src_glob_pattern):
            name = p.name.lower()
            if ("cat" in name) and (p.suffix.lower() in [".jpg", ".jpeg", ".png"]):
                copy2(p, dst_root/"cat"/p.name); moved += 1
            elif ("dog" in name) and (p.suffix.lower() in [".jpg", ".jpeg", ".png"]):
                copy2(p, dst_root/"dog"/p.name); moved += 1
        return moved

    # Case A) DATA_ROOT 바로 아래에 이미지가 잔뜩 있는 경우
    if any(DATA_ROOT.glob("*.jpg")) or any(DATA_ROOT.glob("*.jpeg")) or any(DATA_ROOT.glob("*.png")):
        print("[prep] 데이터가 DATA_ROOT 바로 아래에 섞여 있음 -> train/ 로 정리")
        (train_dir).mkdir(parents=True, exist_ok=True)
        moved = move_by_name(str(DATA_ROOT/"*.*"), train_dir)
        print(f"[prep] train/ 로 복사: {moved}개")

    # Case B) DATA_ROOT/train 아래에 이미지가 섞여 있는 경우
    if train_dir.exists() and (any(train_dir.glob("*.jpg")) or any(train_dir.glob("*.jpeg")) or any(train_dir.glob("*.png"))):
        print("[prep] train/ 아래에 이미지가 섞여 있음 -> cat/, dog/ 로 정리")
        moved = move_by_name(str(train_dir/"*.*"), train_dir)
        print(f"[prep] train/cat, train/dog 로 복사: {moved}개")

    # Case C) val 폴더가 따로 있고 그 안에 섞여 있는 경우도 정리
    if val_dir.exists() and (any(val_dir.glob("*.jpg")) or any(val_dir.glob("*.jpeg")) or any(val_dir.glob("*.png"))):
        print("[prep] val/ 아래에 이미지가 섞여 있음 -> cat/, dog/ 로 정리")
        moved = move_by_name(str(val_dir/"*.*"), val_dir)
        print(f"[prep] val/cat, val/dog 로 복사: {moved}개")

    # 간단한 검증: 파일 열어보기 (깨진 이미지 대비)
    def sanity_check_some(root: Path):
        for cls in ["cat", "dog"]:
            for p in (root/cls).glob("*.*"):
                try:
                    Image.open(p).verify()
                except Exception:
                    print(f"[warn] 손상 이미지 무시: {p}")
                break  # 클래스당 1장만 테스트
    if train_dir.exists():
        sanity_check_some(train_dir)
    if val_dir.exists():
        sanity_check_some(val_dir)

def auto_fetch_if_missing(DATA_ROOT: Path, train_dir: Path, val_dir: Path, seed=SEED, val_ratio=0.1):
    """로컬에 데이터가 없으면 Oxford-IIIT Pet을 받아 cat/dog만 골라 ImageFolder로 구성"""
    has_train = train_dir.exists() and any((train_dir/"cat").glob("*.*")) and any((train_dir/"dog").glob("*.*"))
    has_val   = val_dir.exists()   and any((val_dir/"cat").glob("*.*"))   and any((val_dir/"dog").glob("*.*"))
    if has_train and (has_val or val_ratio > 0):
        return  # 이미 있음

    print("[auto] 로컬 데이터 없음 → Oxford‑IIIT Pet 다운로드…")
    ds = OxfordIIITPet(root=str(DATA_ROOT), split="trainval", target_types="binary-category", download=True)

    items = []
    for i in range(len(ds)):
        path = ds._images[i]
        target = int(ds._labels[i])  # 0=cat, 1=dog
        try:
            Image.open(path).verify()  # 손상 파일 거르기
            items.append((path, "cat" if target == 0 else "dog"))
        except Exception:
            pass

    random.Random(seed).shuffle(items)
    n_total = len(items)
    n_val   = int(n_total * val_ratio)
    val_items = items[:n_val]
    train_items = items[n_val:]

    for p in [train_dir/"cat", train_dir/"dog", val_dir/"cat", val_dir/"dog"]:
        p.mkdir(parents=True, exist_ok=True)

    from shutil import copy2
    def safe_copy(src, dst_dir: Path):
        try:
            copy2(src, dst_dir/src.split(os.sep)[-1])
        except Exception:
            pass

    for src, cls in train_items:
        safe_copy(src, train_dir/cls)
    for src, cls in val_items:
        safe_copy(src, val_dir/cls)

    print(f"[auto] 준비 완료: train={len(train_items)}장, val={len(val_items)}장")


prepare_imagefolder(DATA_ROOT, train_dir, val_dir)
auto_fetch_if_missing(DATA_ROOT, train_dir, val_dir)  # ← 이 줄 추가!


# ====== 실행: 실제로 구조 정리 시도 ======
print(f"[debug] DATA_ROOT = {DATA_ROOT}")
print(f"[debug] train_dir exists: {train_dir.exists()}, cat: {any((train_dir/'cat').glob('*.*'))}, dog: {any((train_dir/'dog').glob('*.*'))}")
print(f"[debug] val_dir   exists: {val_dir.exists()},   cat: {any((val_dir/'cat').glob('*.*'))},   dog: {any((val_dir/'dog').glob('*.*'))}")

# -----------------------
# 1) 변환(전처리)
# -----------------------
IMG_SIZE = 128
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

# -----------------------
# 2) 데이터셋 준비 (val 폴더 없으면 자동 분할)
# -----------------------
has_val = val_dir.exists() and any((val_dir/"cat").glob("*.jpg")) and any((val_dir/"dog").glob("*.jpg"))

if train_dir.exists() and any((train_dir/"cat").glob("*.jpg")) and any((train_dir/"dog").glob("*.jpg")):
    full_train = datasets.ImageFolder(train_dir, transform=train_tf)
    if has_val:
        val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
        train_ds = full_train
    else:
        # train만 있으면 90:10으로 쪼갬
        n_total = len(full_train)
        n_val = int(n_total * 0.1)
        n_train = n_total - n_val
        train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
        # 검증셋 transform 교체(augmentation 제거)
        val_ds.dataset.transform = val_tf
else:
    raise FileNotFoundError(
        f"데이터가 없습니다. 아래 구조로 이미지 준비하세요:\n"
        f"{DATA_ROOT}/train/cat/*.jpg, {DATA_ROOT}/train/dog/*.jpg\n"
        f"옵션) {DATA_ROOT}/val/cat/*.jpg, {DATA_ROOT}/val/dog/*.jpg"
    )

BATCH = 64
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

classes = train_ds.dataset.classes if hasattr(train_ds, "dataset") else train_ds.classes
print(f"classes: {classes}")

# -----------------------
# 3) 아주 쉬운 CNN 모델
# -----------------------
"""
ReLU:
    CNN은 이미지를 숫자로 보고
    계속해서 복잡한 특징을 추출하는데,
    이 과정에서 계산된 값들이 양수/음수 섞여서 나와.

    그런데 음수값은 정보라고 보기 어렵거나, 잡음일 가능성이 많아.

    그래서 ReLU가 이렇게 말해:

    “에이~ 음수는 버려! 의미 없는 거니까 0으로 치자!”

    즉,

    중요하거나 활성화된 정보는 살리고

    덜 중요한 건 무시한다(0으로 만든다)
    """
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 특징 뽑기: "가장자리/점/무늬 같은 로컬 패턴"을 찾음
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 작은 필터로 여러 위치 패턴을 스캔
            nn.BatchNorm2d(16),               # 배치마다 분포 맞춰 학습 안정화
    
            nn.ReLU(),                        # 음수 제거 → "있음/없음" 신호 강조
            nn.MaxPool2d(2),                  # 해상도 절반(64x64) → 요약하고 잡음 줄임

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 32x32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 16x16
        )
        # 분류기: "뽑아낸 특징"을 가지고 고양이/개 확률로 압축
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)  # 최종 로짓(softmax 직전 값)
        )

    def forward(self, x):
        x = self.features(x)   # (B, 64, 16, 16)
        x = self.classifier(x) # (B, 2)
        return x

model = SimpleCNN(num_classes=2).to(DEVICE)
print(model)

# -----------------------
# 4) (교육모드) 훅/시각화
# -----------------------
EDU_MODE = True   # ← False로 두면 조용히 학습만 함
MAX_CH = 16       # feature map 그릴 때 채널 수 제한(보기 편하게)

import matplotlib.pyplot as plt

def _minmax(t):
    t = t.detach()
    return float(t.min().cpu()), float(t.max().cpu())

def show_feature_map(t_3chw, title="Feature Map", max_channels=MAX_CH, ncols=8):
    """(C,H,W) 텐서를 채널별 흑백 이미지로 출력"""
    t = t_3chw.detach().cpu()
    C, H, W = t.shape
    C = min(C, max_channels)
    nrows = (C + ncols - 1) // ncols
    plt.figure(figsize=(ncols*2, nrows*2))
    for i in range(C):
        ch = t[i]
        # 보기 좋게 채널별 정규화
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        ax = plt.subplot(nrows, ncols, i+1)
        ax.imshow(ch.numpy(), cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# 블록별로 "무슨 역할인지"를 로그로 찍고, 중간 출력 일부 이미지를 보여준다.
def explain_and_visualize_one(xb_cpu):
    model.eval()
    with torch.no_grad():
        x = xb_cpu.to(DEVICE)

        # Block1
        conv1 = model.features[0](x); bn1 = model.features[1](conv1); relu1 = model.features[2](bn1); pool1 = model.features[3](relu1)
        print("[Block1] Conv→BN→ReLU→Pool")
        print("  Conv1:", conv1.shape, "minmax=", _minmax(conv1[0])); 
        print("   BN1 :", bn1.shape,   "minmax=", _minmax(bn1[0]));
        print("  ReLU1:", relu1.shape, "minmax=", _minmax(relu1[0]));
        print("  Pool1:", pool1.shape, "minmax=", _minmax(pool1[0]), "← 해상도 절반으로, 노이즈 감소/요약")

        # Block2
        conv2 = model.features[4](pool1); bn2 = model.features[5](conv2); relu2 = model.features[6](bn2); pool2 = model.features[7](relu2)
        print("[Block2] Conv→BN→ReLU→Pool (더 복잡한 패턴)")
        print("  Conv2:", conv2.shape, "minmax=", _minmax(conv2[0]));
        print("   BN2 :", bn2.shape,   "minmax=", _minmax(bn2[0]));
        print("  ReLU2:", relu2.shape, "minmax=", _minmax(relu2[0]));
        print("  Pool2:", pool2.shape, "minmax=", _minmax(pool2[0]))

        # Block3
        conv3 = model.features[8](pool2); bn3 = model.features[9](conv3); relu3 = model.features[10](bn3); pool3 = model.features[11](relu3)
        print("[Block3] Conv→BN→ReLU→Pool (귀/코/수염 같은 조합 특징)")
        print("  Conv3:", conv3.shape, "minmax=", _minmax(conv3[0]));
        print("   BN3 :", bn3.shape,   "minmax=", _minmax(bn3[0]));
        print("  ReLU3:", relu3.shape, "minmax=", _minmax(relu3[0]));
        print("  Pool3:", pool3.shape, "minmax=", _minmax(pool3[0]), "→ 마지막 요약본")

        # 분류기 앞/뒤
        flat = model.classifier[0](pool3); h1 = model.classifier[1](flat); h1a = model.classifier[2](h1); out = model.classifier[4](h1a)
        print("[Classifier] Flatten→Linear(128)→ReLU→Dropout→Linear(2)")
        print("  Flatten:", flat.shape)
        print("  Hidden :", h1.shape, "ReLU 후:", h1a.shape)
        print("  Logits :", out.shape, "예: ", out[0].detach().cpu().tolist(), " (softmax 전 점수)")

        # === 시각화(배치 첫 이미지 기준) ===
        img0 = xb_cpu[0]  # (3,H,W)
        # 입력 복원(정규화 해제해서 보기)
        inv = img0.clone()
        inv = inv * torch.tensor([[0.5],[0.5],[0.5]]) + torch.tensor([[0.5],[0.5],[0.5]])
        inv = inv.clamp(0,1).permute(1,2,0).numpy()

        import matplotlib.pyplot as plt
        plt.figure(figsize=(3,3)); plt.imshow(inv); plt.title("입력 이미지(복원)"); plt.axis("off"); plt.show()

        show_feature_map(conv1[0].detach().cpu(), "Conv1 출력: 가장자리/점/단순무늬")
        show_feature_map(pool1[0].detach().cpu(), "Pool1 출력: 해상도↓, 요약")

        show_feature_map(conv2[0].detach().cpu(), "Conv2 출력: 더 복잡한 패턴")
        show_feature_map(conv3[0].detach().cpu(), "Conv3 출력: 조합 특징")

# -----------------------
# 5) 학습 준비
# -----------------------
criterion = nn.CrossEntropyLoss()               # 내부적으로 LogSoftmax 포함 (즉, 학습 시 Softmax 필요없음)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

# -----------------------
# 6) 학습 루프
# -----------------------
EPOCHS = 5
for epoch in range(1, EPOCHS+1):
    # ---- Train ----
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_acc  += accuracy(logits, yb) * xb.size(0)
        n += xb.size(0)
    train_loss = total_loss / n
    train_acc  = total_acc / n

    # ---- Validate ----
    model.eval()
    v_loss, v_acc, vn = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            v_loss += loss.item() * xb.size(0)
            v_acc  += accuracy(logits, yb) * xb.size(0)
            vn += xb.size(0)
    print(f"[{epoch}/{EPOCHS}] train_loss={train_loss:.4f} acc={train_acc:.3f} | val_loss={v_loss/vn:.4f} acc={v_acc/vn:.3f}")

    # ---- (교육모드) 첫/마지막 에폭에 '한 배치'로 설명+그림 ----
    if EDU_MODE and (epoch == 1 or epoch == EPOCHS):
        xb_vis, _ = next(iter(val_loader))
        # CPU 텐서를 전달(시각화 함수 내부에서 .to(DEVICE))
        explain_and_visualize_one(xb_vis[:1].cpu())  # 한 장만 보기

# -----------------------
# 7) 예측(Softmax 확률)
# -----------------------
softmax = nn.Softmax(dim=1)

def predict_image(img_path):
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    x = val_tf(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = softmax(logits).squeeze().cpu().tolist()  # [p_cat, p_dog] (classes 순서)
        idx = int(torch.argmax(logits, dim=1))
        name = classes[idx]
    return name, probs

# 사용 예:
# name, p = predict_image(f"{DATA_ROOT}/train/cat/xxx.jpg")
# print("예측:", name, "확률[cat,dog]=", [round(v,3) for v in p])
