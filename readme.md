# Python 3.10 가상환경

uv python install 3.10
uv venv -p 3.10

# 활성화 (Windows PowerShell)

. .venv/Scripts/Activate.ps1

# macOS/Linux

# source .venv/bin/activate

uv pip install -r requirements.txt

# gpu 있는 local pc에서 할때

uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
