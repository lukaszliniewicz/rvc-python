#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd -- "$PROJECT_DIR/.." && pwd)"
PIXI_EXE="$PARENT_DIR/bin/pixi"
PIXI_VERSION="0.68.1"
PIXI_SHA256="01d29d4b78ab07badf57edda0b3d200bc705d5afb6da9960ebabe7010cd836e4"
BACKEND="auto"
PASS_ARGS=()

while (($#)); do
    case "$1" in
        --pixi-path)
            [[ $# -ge 2 ]] || { echo "--pixi-path requires a value" >&2; exit 2; }
            PIXI_EXE="$2"
            shift 2
            ;;
        --pixi-path=*)
            PIXI_EXE="${1#*=}"
            shift
            ;;
        --backend)
            [[ $# -ge 2 ]] || { echo "--backend requires a value" >&2; exit 2; }
            BACKEND="${2,,}"
            shift 2
            ;;
        --backend=*)
            BACKEND="${1#*=}"
            BACKEND="${BACKEND,,}"
            shift
            ;;
        *)
            PASS_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$(uname -m)" in
    x86_64|amd64) ;;
    *) echo "RVC on Linux currently requires x86_64." >&2; exit 2 ;;
esac

if [[ "$BACKEND" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        BACKEND="cuda"
    else
        BACKEND="cpu"
    fi
fi

case "$BACKEND" in
    cpu) PIXI_ENV="cpu" ;;
    cuda) PIXI_ENV="default" ;;
    *) echo "Unsupported RVC backend: $BACKEND" >&2; exit 2 ;;
esac

if [[ ! -x "$PIXI_EXE" ]]; then
    mkdir -p -- "$(dirname -- "$PIXI_EXE")"
    PIXI_URL="https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl"
    PIXI_DOWNLOAD="${PIXI_EXE}.download"
    rm -f -- "$PIXI_DOWNLOAD"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --output "$PIXI_DOWNLOAD" "$PIXI_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget --output-document="$PIXI_DOWNLOAD" "$PIXI_URL"
    else
        echo "Pixi is missing and neither curl nor wget is available." >&2
        exit 2
    fi
    if ! echo "$PIXI_SHA256  $PIXI_DOWNLOAD" | sha256sum --check --status; then
        rm -f -- "$PIXI_DOWNLOAD"
        echo "Downloaded Pixi binary failed SHA-256 verification." >&2
        exit 2
    fi
    mv -- "$PIXI_DOWNLOAD" "$PIXI_EXE"
    chmod 0755 "$PIXI_EXE"
fi

export PIXI_CACHE_DIR="$PARENT_DIR/.pixi-cache"
export PIP_CACHE_DIR="$PARENT_DIR/.pip-cache"
export TMPDIR="$PARENT_DIR/.tmp"
export PIXI_FROZEN=true
mkdir -p -- "$PIXI_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

cd -- "$PROJECT_DIR"
"$PIXI_EXE" install --frozen --environment "$PIXI_ENV"
# Keep this shell as the stable root process for supervisors such as Pandrator
# Manager. Replacing it with Pixi via `exec` changes the recorded executable at
# the same PID and can be mistaken for PID reuse by conservative supervisors.
"$PIXI_EXE" run --environment "$PIXI_ENV" python run.py --backend "$BACKEND" "${PASS_ARGS[@]}"
