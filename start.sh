#!/usr/bin/env bash
# ============================================================
# ACG — AnimateDiff / Wan 2.2 T2V start.sh
# Compatible with RunPod on-demand GPU pods.
#
# Node repos (from workflow JSON):
#   - VHS         → Kosinkadink/ComfyUI-VideoHelperSuite
#   - FILM VFI    → Fannovel16/ComfyUI-Frame-Interpolation
#   - KJNodes     → kijai/ComfyUI-KJNodes (SageAttention patch, ImageResizeKJ)
#   - RES4LYF     → ClownsharkBatwing/RES4LYF (ClownsharKSampler_Beta)
#   - rgthree     → rgthree/rgthree-comfy (Fast Groups Bypasser)
#
# Models (from workflow JSON):
#   UNet   : wan2.2_t2v_low_noise_14B_fp16.safetensors  (Comfy-Org HF)
#   VAE    : wan_2.1_vae.safetensors                     (Comfy-Org HF)
#   CLIP   : umt5_xxl_fp8_e4m3fn_scaled.safetensors     (Comfy-Org HF)
#   LoRA   : lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16 (Kijai HF)
#   FILM   : film_net_fp32.pt                            (Fannovel16 GH release)
#   LoRA   : CHAR_LORA_URL env var (character-specific, set in pod template)
# ============================================================
set -euo pipefail

echo "==================================================="
echo " ACG AnimateDiff / Wan 2.2 Pod — Starting"
echo "==================================================="

# ─────────────────────────────────────────────────────────────
# 0. PATH ALIASES
# ─────────────────────────────────────────────────────────────
COMFY_DIR="${COMFYUI_PATH:-/workspace/ComfyUI}"
CUSTOM_NODES="${COMFY_DIR}/custom_nodes"
MODELS_DIR="${COMFY_DIR}/models"
BAKED_DIR="${COMFYUI_BAKED:-/opt/ComfyUI}"

PERSIST_DIR="${RUNPOD_VOLUME:-/workspace/runpod-volume}"
REPO_CACHE="${PERSIST_DIR}/_repos"
PIP_CACHE_DIR="${PERSIST_DIR}/.cache/pip"

UPDATE_NODES="${UPDATE_NODES:-0}"
INSTALL_NODE_REQS="${INSTALL_NODE_REQS:-1}"

mkdir -p "$PERSIST_DIR" "$REPO_CACHE" "$PIP_CACHE_DIR"
export PIP_CACHE_DIR PIP_DISABLE_PIP_VERSION_CHECK=1

# ─────────────────────────────────────────────────────────────
# 1. NETWORK READINESS
# ─────────────────────────────────────────────────────────────
echo "[net] Waiting for network..."
MAX_WAIT=30; waited=0
while [ $waited -lt $MAX_WAIT ]; do
  if ping -c1 -W2 8.8.8.8 >/dev/null 2>&1; then
    echo "[net] Ready (${waited}s)"; break
  fi
  waited=$((waited + 1)); sleep 1
done
[ $waited -ge $MAX_WAIT ] && echo "[net] WARNING: network may not be ready"

# ─────────────────────────────────────────────────────────────
# 2. COMFYUI RESTORE
# ─────────────────────────────────────────────────────────────
if [ ! -f "${COMFY_DIR}/main.py" ] && [ -f "${BAKED_DIR}/main.py" ]; then
  echo "[setup] Restoring ComfyUI from baked image..."
  cp -a "${BAKED_DIR}" "${COMFY_DIR}"
fi
[ ! -f "${COMFY_DIR}/main.py" ] && echo "[FATAL] ComfyUI not found" && exit 1

mkdir -p "${CUSTOM_NODES}" "${MODELS_DIR}"

# ─────────────────────────────────────────────────────────────
# 3. PIP CONSTRAINTS
# ─────────────────────────────────────────────────────────────
CONSTRAINTS_FILE="${PERSIST_DIR}/pip-constraints.txt"
cat > "$CONSTRAINTS_FILE" <<'EOF'
numpy<2
protobuf<5
transformers>=4.45.0
safetensors
mediapipe==0.10.14
sageattention
EOF
export PIP_CONSTRAINT="$CONSTRAINTS_FILE"

# ─────────────────────────────────────────────────────────────
# 4. HELPERS
# ─────────────────────────────────────────────────────────────

download() {
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  [ -f "$out" ] && [ -s "$out" ] && { echo "[dl] exists: $(basename "$out")"; return 0; }
  echo "[dl] → $(basename "$out")"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x16 -s16 -k1M \
      --allow-overwrite=true --file-allocation=none \
      --max-tries=5 --retry-wait=3 \
      --connect-timeout=30 --timeout=300 \
      --max-connection-per-server=16 --min-split-size=1M \
      --stream-piece-selector=geom --optimize-concurrent-downloads=true \
      -d "$(dirname "$out")" -o "$(basename "$out")" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -fL --retry 8 --retry-delay 2 -C - -o "$out" "$url"
  else
    wget -c -O "$out" "$url"
  fi
}

civit_download() {
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  [ -f "$out" ] && [ -s "$out" ] && { echo "[civit] exists: $(basename "$out")"; return 0; }
  echo "[civit] → $(basename "$out")"
  local dl_url="$url"
  local ua="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
  if [ -n "${CIVITAI_TOKEN:-}" ]; then
    dl_url=$(curl -sL -I -A "$ua" \
      -H "Authorization: Bearer ${CIVITAI_TOKEN}" \
      "$url" | grep -i "^location:" | tail -1 | sed 's/^location: //i' | tr -d '\r\n')
    [ -z "$dl_url" ] && dl_url="$url"
  fi
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x16 -s16 -k1M \
      --allow-overwrite=true --file-allocation=none \
      --max-tries=10 --retry-wait=2 \
      --connect-timeout=30 --timeout=300 \
      --max-connection-per-server=16 --min-split-size=1M \
      --user-agent="$ua" \
      -d "$(dirname "$out")" -o "$(basename "$out")" "$dl_url"
  else
    local auth_header=()
    [ -n "${CIVITAI_TOKEN:-}" ] && auth_header=(-H "Authorization: Bearer ${CIVITAI_TOKEN}")
    curl -fL --retry 10 --retry-delay 2 -C - -A "$ua" "${auth_header[@]}" -o "$out" "$dl_url"
  fi
  if command -v file >/dev/null 2>&1 && file "$out" | grep -qi "HTML"; then
    echo "[civit] ERROR: got HTML (token missing/gated). Removing."
    rm -f "$out"; return 1
  fi
}

env_lora_download() {
  local var="$1" filename="${2:-}"
  local url="${!var:-}"
  [ -z "$url" ] && { echo "[lora] ${var} not set → skip"; return 0; }
  [ -z "$filename" ] && filename="$(basename "${url%%\?*}")"
  filename="${filename// /_}"
  download "$url" "${MODELS_DIR}/loras/${filename}"
}

safe_pip_req() {
  local req="$1"
  [ -f "$req" ] || return 0
  local tmp; tmp="$(mktemp)"
  grep -viE '^(torch|torchvision|torchaudio|numpy|transformers|tokenizers|protobuf)([<=>].*)?$' "$req" > "$tmp" || true
  local delay=2
  for i in 1 2 3; do
    pip install -q --prefer-binary --retries 5 --timeout 60 \
      -c "$CONSTRAINTS_FILE" -r "$tmp" && break
    echo "  [pip] retry $i/3 in ${delay}s..."; sleep $delay; delay=$((delay * 2))
  done
  rm -f "$tmp"
}

git_cache() {
  local repo="$1" dir="$2" update_flag="$3"
  if [ ! -d "${dir}/.git" ]; then
    echo "[git] cloning $(basename "$repo")..."
    GIT_TERMINAL_PROMPT=0 git clone --depth 1 --progress "$repo" "$dir"
  elif [ "${!update_flag:-0}" = "1" ]; then
    echo "[git] updating $(basename "$dir")..."
    git -C "$dir" fetch --depth=1 origin main
    git -C "$dir" reset --hard origin/main || true
  else
    echo "[git] cached: $(basename "$dir")"
  fi
}

link_node_pack() {
  local repo_dir="$1"
  local has_nodes=0
  for dir in "$repo_dir"/*/; do
    [ -d "$dir" ] || continue
    local name; name="$(basename "$dir")"
    case "$name" in .git|.github|__pycache__|docs|examples|tests) continue ;; esac
    { [ -f "${dir}/__init__.py" ] || [ -f "${dir}/nodes.py" ]; } || continue
    ln -sfn "$dir" "${CUSTOM_NODES}/${name}"
    has_nodes=1
  done
  if [ "$has_nodes" = "0" ] && { [ -f "${repo_dir}/__init__.py" ] || [ -f "${repo_dir}/nodes.py" ]; }; then
    ln -sfn "$repo_dir" "${CUSTOM_NODES}/$(basename "$repo_dir")"
  fi
}

# ─────────────────────────────────────────────────────────────
# 5. SECRETS MAPPING
# ─────────────────────────────────────────────────────────────
if [ -z "${CIVITAI_TOKEN:-}" ] && [ -n "${RUNPOD_SECRET_CivitKey:-}" ]; then
  export CIVITAI_TOKEN="${RUNPOD_SECRET_CivitKey}"
  echo "[config] CivitAI token loaded from RunPod secret"
fi

# ─────────────────────────────────────────────────────────────
# 6. CUSTOM NODE REPOS — clone/update in parallel
# ─────────────────────────────────────────────────────────────
echo "[repos] Setting up custom node repos..."

VHS_REPO="${REPO_CACHE}/ComfyUI-VideoHelperSuite"
FILM_REPO="${REPO_CACHE}/ComfyUI-Frame-Interpolation"
KJNODES_REPO="${REPO_CACHE}/ComfyUI-KJNodes"
RES4LYF_REPO="${REPO_CACHE}/RES4LYF"
RGTHREE_REPO="${REPO_CACHE}/rgthree-comfy"

(git_cache "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"   "$VHS_REPO"      UPDATE_NODES) &
(git_cache "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git" "$FILM_REPO"     UPDATE_NODES) &
(git_cache "https://github.com/kijai/ComfyUI-KJNodes.git"                  "$KJNODES_REPO"  UPDATE_NODES) &
(git_cache "https://github.com/ClownsharkBatwing/RES4LYF.git"              "$RES4LYF_REPO"  UPDATE_NODES) &
(git_cache "https://github.com/rgthree/rgthree-comfy.git"                  "$RGTHREE_REPO"  UPDATE_NODES) &

wait
echo "[repos] All repos ready"

# ─────────────────────────────────────────────────────────────
# 7. SYMLINK REPOS INTO custom_nodes
# ─────────────────────────────────────────────────────────────
for repo in "$VHS_REPO" "$FILM_REPO" "$KJNODES_REPO" "$RES4LYF_REPO" "$RGTHREE_REPO"; do
  link_node_pack "$repo"
done

# ─────────────────────────────────────────────────────────────
# 8. MODEL DIRECTORIES
# ─────────────────────────────────────────────────────────────
mkdir -p \
  "${MODELS_DIR}/diffusion_models" \
  "${MODELS_DIR}/vae" \
  "${MODELS_DIR}/clip" \
  "${MODELS_DIR}/loras" \
  "${MODELS_DIR}/FILM"

# ─────────────────────────────────────────────────────────────
# 9. MODEL DOWNLOADS — all in parallel
# ─────────────────────────────────────────────────────────────
echo "[models] Starting parallel downloads..."

# ── Wan 2.2 T2V UNet — fp16 (28.6 GB)
# Swap to fp8_scaled variant (~14 GB) if VRAM limited:
# wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors
download \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors" \
  "${MODELS_DIR}/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors" &

# ── VAE
download \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
  "${MODELS_DIR}/vae/wan_2.1_vae.safetensors" &

# ── CLIP / Text Encoder (6.74 GB)
# Workflow loads from models/clip/ — ComfyUI checks both clip/ and text_encoders/
download \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
  "${MODELS_DIR}/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors" &

# ── lightx2v distill LoRA rank256 bf16 (Kijai HF)
# Exact filename match from workflow: lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16
download \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors" \
  "${MODELS_DIR}/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors" &

# ── FILM VFI model (GitHub release)
# ComfyUI-Frame-Interpolation will also auto-download this on first run,
# but pre-downloading avoids timeout failures on RunPod cold start.
download \
  "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/v1.0.0/film_net_fp32.pt" \
  "${MODELS_DIR}/FILM/film_net_fp32.pt" &

# ── Character LoRA (set CHAR_LORA_URL in RunPod pod template env vars)
env_lora_download "CHAR_LORA_URL" &

wait
echo "[models] All downloads complete"

# ─────────────────────────────────────────────────────────────
# 10. NODE REQUIREMENTS
# ─────────────────────────────────────────────────────────────
REQ_MARK="${PERSIST_DIR}/.node-reqs-ok"

if [ "$INSTALL_NODE_REQS" = "1" ] && { [ ! -f "$REQ_MARK" ] || [ "$UPDATE_NODES" = "1" ]; }; then
  echo "[pip] Installing node requirements..."
  for repo in "$VHS_REPO" "$FILM_REPO" "$KJNODES_REPO" "$RES4LYF_REPO" "$RGTHREE_REPO"; do
    for dir in "$repo"/*/; do
      safe_pip_req "${dir}/requirements.txt"
    done
    safe_pip_req "${repo}/requirements.txt"
  done
  touch "$REQ_MARK"
  echo "[pip] Node requirements done"
else
  echo "[pip] Node requirements already installed (skip)"
fi

# ─────────────────────────────────────────────────────────────
# 11. FINAL SANITY
# ─────────────────────────────────────────────────────────────
echo "[pip] Re-pinning critical packages..."
pip install -q --prefer-binary --retries 5 --timeout 60 \
  -c "$CONSTRAINTS_FILE" "numpy<2" "mediapipe==0.10.14" || \
  echo "[pip] WARNING: Re-pin failed — proceeding"

# ─────────────────────────────────────────────────────────────
# 12. VERSION REPORT
# ─────────────────────────────────────────────────────────────
python3 - <<'PY'
import sys, torch
print(f"python   : {sys.version.split()[0]}")
print(f"torch    : {torch.__version__}  cuda={torch.version.cuda}")
try:
    import numpy; print(f"numpy    : {numpy.__version__}")
    import transformers; print(f"transformers: {transformers.__version__}")
    import mediapipe; print(f"mediapipe: {mediapipe.__version__}")
except Exception as e:
    print(f"[warn] {e}")
PY

# ─────────────────────────────────────────────────────────────
# 13. JUPYTERLAB (background)
# ─────────────────────────────────────────────────────────────
echo "[jupyter] Starting JupyterLab on :8888..."
jupyter lab \
  --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
  --ServerApp.token='' --ServerApp.password='' \
  --ServerApp.allow_origin='*' \
  --ServerApp.root_dir="${COMFY_DIR}" \
  >/workspace/jupyter.log 2>&1 &

# ─────────────────────────────────────────────────────────────
# 14. LAUNCH COMFYUI
# ─────────────────────────────────────────────────────────────
echo "==================================================="
echo " Launching ComfyUI on :8188"
echo "==================================================="
cd "${COMFY_DIR}"
exec python3 main.py --listen 0.0.0.0 --port 8188
