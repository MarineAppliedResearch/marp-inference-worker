#!/bin/sh
# bootstrap-linux.sh
# Created: 2026-09-18
# Author: Isaac Travers
#
# Turns a Linux computer into a MARP volunteer worker. An NVIDIA GPU is preferred
# and not required -- without one it installs the CPU build and works slowly, which
# is worth more than turning the volunteer away.
#
# The POSIX counterpart of bootstrap-windows.ps1, and deliberately the same shape:
# the same stages in the same order, the same lock files with url/size/sha256, the
# same refusal to proceed on an unverified download. Where the two differ, it is
# because the platform forced it, and each of those is commented.
#
# This is a *downloader*, not a bundle. It carries locks, the worker source and the
# enrolment code; it fetches the Python runtime, the wheels and Chromium. That was
# chosen deliberately: bundling would mean MARP serving several gigabytes per
# volunteer per release, where this serves a few megabytes and lets PyPI, Astral
# and Google serve the rest.
#
# /bin/sh rather than bash: a volunteer's machine may not have bash, and nothing
# here needs it.

set -eu

# ---------------------------------------------------------------------------
# Build-time inputs. `build-linux.sh` replaces these at package time; they are
# never committed with real values. The enrolment code in particular must not
# reach the repository -- MARP_API#208 makes that a requirement rather than a
# preference.
# ---------------------------------------------------------------------------
COORDINATOR_URL="${MARP_COORDINATOR_URL:-@MARP_COORDINATOR_URL@}"
ENROLMENT_CODE="${MARP_ACTIVATION_CODE:-@MARP_ENROLMENT_CODE@}"

# Everything lives under one directory so uninstalling is `rm -rf`, and so nothing
# needs root. A volunteer should never be asked for a password to donate compute.
INSTALL_ROOT="${MARP_INSTALL_ROOT:-$HOME/.local/share/marp-worker}"
STATE_DIR="$INSTALL_ROOT/state"
VENV="$INSTALL_ROOT/venv"
DOWNLOADS="$INSTALL_ROOT/downloads"
LOCKS="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
LOG="$INSTALL_ROOT/install.log"

# Enough for the venv (~7 GB), Chromium (~600 MB unpacked) and headroom.
# The CUDA build needs about 8 GB installed; the CPU build about 2 GB. Checked
# against whichever is actually being installed rather than always demanding the
# larger, so a small CPU-only machine is not turned away for space it will not use.
REQUIRED_FREE_CUDA_MB=12000
REQUIRED_FREE_CPU_MB=4000

TOTAL_STAGES=8
stage_number=0

# --check-only answers "will this work on my computer?" without committing to a
# multi-gigabyte download. Worth having for its own sake: a volunteer whose driver
# is missing should find out in two seconds rather than after twenty minutes of
# downloading, and it is the difference between them fixing it and giving up.
CHECK_ONLY=no
for arg in "$@"; do
    case "$arg" in
        --check-only) CHECK_ONLY=yes ;;
        --help|-h)
            printf '%s\n' \
                "MARP volunteer worker setup" \
                "" \
                "  (no options)   install and start the worker" \
                "  --check-only   check this computer is suitable, change nothing" \
                "  --help         this message" \
                "" \
                "Environment overrides:" \
                "  MARP_INSTALL_ROOT   where to install (default ~/.local/share/marp-worker)"
            exit 0 ;;
        *) printf 'Unknown option: %s (try --help)\n' "$arg" >&2; exit 2 ;;
    esac
done


# say()
# One line to the volunteer and to the log.
# Inputs: the message.
# Output: none.
say() {
    printf '%s\n' "$*"
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG" 2>/dev/null || true
}


# stage()
# Announce a numbered stage, so a volunteer watching a long install can see it is
# progressing rather than hung. The Windows script does the same.
stage() {
    stage_number=$((stage_number + 1))
    say ""
    say "[$stage_number/$TOTAL_STAGES] $*"
}


# fail()
# Stop with a message a volunteer can act on.
# Inputs: the explanation, then optionally what to do about it.
# Output: exits 1.
#
# Every failure path in this script goes through here. An installer that dies on an
# unhandled error teaches the volunteer that the project is broken, when usually
# they are one missing thing away from working.
fail() {
    # Snapshot the log before writing anything, or the tail below shows this
    # function's own output instead of the error that caused it.
    # Only past stage 1. A preflight refusal explains itself in full and its log
    # tail is just the banner, which buries the actionable message in noise. A
    # failure inside a real step is the opposite: the useful detail is whatever the
    # tool printed, and it is the only place it exists.
    failure_context=""
    [ "$stage_number" -gt 1 ] && [ -s "$LOG" ] && failure_context=$(tail -n 10 "$LOG" 2>/dev/null)

    say ""
    say "MARP setup stopped."
    say ""
    for line in "$@"; do say "  $line"; done
    # Show what the failing step actually said. Pointing at a log file is useless
    # advice on a volunteer's machine: they have to find it, open it, and know
    # which part matters -- and if they are reporting the problem to us, the lines
    # we need are the ones we just told them to go and look for themselves.
    if [ -n "$failure_context" ]; then
        say "  What the last step reported:"
        say ""
        printf '%s\n' "$failure_context" | sed 's/^/      /'
        say ""
    fi
    say "  Nothing was left running. Fix the above and run this installer again --"
    say "  it continues from where it stopped rather than starting over."
    say ""
    say "  Full log: $LOG"
    exit 1
}


# need()
# Assert a command exists, with a distribution-neutral hint when it does not.
need() {
    command -v "$1" >/dev/null 2>&1 || fail \
        "This installer needs '$1', which is not on this computer." \
        "Install it with your system's package manager, then run this again." \
        "  Debian/Ubuntu:  sudo apt install $2" \
        "  Fedora:         sudo dnf install $2" \
        "  Arch:           sudo pacman -S $2"
}


# fetch()
# Download a locked artifact and refuse anything that does not match its hash.
# Inputs: the lock file name, and the destination path.
# Output: none; the file exists and is verified, or the script exits.
#
# Resumable, because a volunteer on a domestic connection downloading 236 MB of
# Chromium will sometimes lose it halfway, and starting over is how people give up.
# Verified, because an unverified download is an arbitrary-code-execution bug with
# extra steps.
fetch() {
    lock="$LOCKS/$1"
    dest="$2"
    [ -f "$lock" ] || fail "Missing lock file: $1" "This installer package is incomplete."

    url=$(sed -n 's/.*"url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$lock")
    want=$(sed -n 's/.*"sha256"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$lock")
    name=$(sed -n 's/.*"name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$lock")

    [ -n "$url" ] && [ -n "$want" ] || fail "Lock file $1 is malformed."

    # Already have it and it verifies: nothing to do. This is what makes re-running
    # after a failure cheap instead of a full re-download.
    if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$want" ]; then
        say "    $name already downloaded and verified"
        return 0
    fi

    attempt=1
    while [ "$attempt" -le 3 ]; do
        say "    downloading $name (attempt $attempt of 3)"
        # -C - resumes a partial file; --retry covers transient DNS and 5xx.
        if curl -fL -C - --retry 3 --retry-delay 2 --progress-bar -o "$dest" "$url"; then
            got=$(sha256sum "$dest" | cut -d' ' -f1)
            if [ "$got" = "$want" ]; then
                say "    $name verified"
                return 0
            fi
            # A resumed download onto a corrupt partial file can never verify, so
            # throw it away rather than resuming onto the same wreckage.
            say "    checksum mismatch, discarding and retrying"
            rm -f "$dest"
        fi
        attempt=$((attempt + 1))
    done

    fail "Could not download $name after 3 attempts." \
         "Check this computer's internet connection and run the installer again."
}


mkdir -p "$INSTALL_ROOT" "$DOWNLOADS" "$STATE_DIR"

say "MARP volunteer worker setup"
say "Installing into $INSTALL_ROOT"
say "Nothing here needs administrator rights."

# ---------------------------------------------------------------------------
stage "Checking this computer..."
# ---------------------------------------------------------------------------
need curl curl
need sha256sum coreutils
need tar tar
need unzip unzip

# A C compiler, because `cython-bbox` publishes no wheel for any platform and is
# built from source during stage 5. Checked here rather than discovered there:
# without this the install fails after several gigabytes have downloaded, which is
# the most expensive moment to find out and the one most likely to lose a
# volunteer. ByteTrack needs it and ByteTrack is not optional -- the tracking stage
# imports BYTETracker from the vendored tree.
command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 || fail \
    "This installer needs a C compiler, which is not on this computer." \
    "One of MARP's components has no prebuilt package and is compiled during setup." \
    "" \
    "  Debian/Ubuntu:  sudo apt install build-essential" \
    "  Fedora:         sudo dnf groupinstall \"Development Tools\"" \
    "  Arch:           sudo pacman -S base-devel" \
    "" \
    "  Then run this installer again."

# A GPU is preferred and not required. A machine without one still contributes,
# slowly, so this warns and continues rather than refusing -- "any computer, any
# GPU, or no GPU" is the goal, and an installer that turns people away at the door
# is the main thing that stopped it being true.
#
# A GPU driver is a kernel module and no user-space installer can ship one, so this
# is the one dependency the package cannot carry on any platform. All it can do is
# say so clearly.
CAPABILITY=""
GPU_NAME=""
if command -v nvidia-smi >/dev/null 2>&1; then
    CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
fi

if [ -n "$CAPABILITY" ]; then
    say "    GPU: $GPU_NAME (compute capability $CAPABILITY)"
else
    GPU_MODE=cpu
    say ""
    if command -v nvidia-smi >/dev/null 2>&1; then
        say "    An NVIDIA driver is present but no GPU answered."
        say "    This usually means the driver was updated without restarting."
    else
        say "    No NVIDIA graphics card was found."
    fi
    say ""
    say "    MARP will install and run on the processor instead. That works, and it"
    say "    is roughly twenty times slower than a graphics card -- so this computer"
    say "    will contribute, but slowly."
    say ""
    say "    To use a graphics card instead, install its driver, restart, and run"
    say "    this installer again:"
    say "      Ubuntu/Debian:  sudo ubuntu-drivers autoinstall"
    say "      Fedora:         enable RPM Fusion, then sudo dnf install akmod-nvidia"
    say "      Arch:           sudo pacman -S nvidia"
    say ""
fi

# Same rule as bootstrap-windows.ps1:92-102, and it must stay the same rule:
# consumer Blackwell reports 12.x and needs kernels built with CUDA 12.8. Without a
# card we take the CPU build, which is also about 6 GB smaller -- a machine that
# cannot use the CUDA libraries should not download them.
if [ -z "$CAPABILITY" ]; then
    TORCH_BACKEND=cpu
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    say "    selected runtime: cpu"
else
    CUDA_MAJOR=${CAPABILITY%%.*}
    if [ "$CUDA_MAJOR" -ge 12 ] 2>/dev/null; then
        TORCH_BACKEND=cu128
    else
        TORCH_BACKEND=cu126
    fi
    TORCH_INDEX="https://download.pytorch.org/whl/$TORCH_BACKEND"
    say "    selected CUDA runtime: $TORCH_BACKEND"
fi

# A driver floor, because the failure without one is silent and expensive: the CUDA
# wheels install perfectly against a too-old driver, and the worker then starts,
# enrols, polls, heartbeats and fails every job -- while reporting itself online.
# Better to refuse here with a version number the volunteer can act on.
#
# 525.60.13 is NVIDIA's minimum for the CUDA 12.x runtime on Linux. cu130 would need
# 580, and is not selected by this installer.
case "$TORCH_BACKEND" in
    cpu)         DRIVER_FLOOR=0 ;;
    cu126|cu128) DRIVER_FLOOR=525 ;;
    *)           DRIVER_FLOOR=580 ;;
esac

DRIVER_VERSION=""
[ "$DRIVER_FLOOR" -gt 0 ] && DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
DRIVER_MAJOR=${DRIVER_VERSION%%.*}

if [ -n "$DRIVER_MAJOR" ] && [ "$DRIVER_MAJOR" -lt "$DRIVER_FLOOR" ] 2>/dev/null; then
    fail \
        "This computer's NVIDIA driver is too old for MARP." \
        "  installed: $DRIVER_VERSION" \
        "  needed:    $DRIVER_FLOOR or newer" \
        "" \
        "  MARP would install successfully and then fail every job, so it stops here" \
        "  rather than looking like it is working." \
        "" \
        "  Ubuntu/Debian:  sudo ubuntu-drivers autoinstall   (then restart)" \
        "  Fedora:         sudo dnf upgrade akmod-nvidia" \
        "  Arch:           sudo pacman -Syu nvidia"
fi
[ -n "$DRIVER_VERSION" ] && say "    driver: $DRIVER_VERSION (needs $DRIVER_FLOOR or newer)"

FREE_MB=$(df -Pm "$INSTALL_ROOT" | awk 'NR==2 {print $4}')
if [ "$TORCH_BACKEND" = cpu ]; then
    REQUIRED_FREE_MB=$REQUIRED_FREE_CPU_MB
else
    REQUIRED_FREE_MB=$REQUIRED_FREE_CUDA_MB
fi
[ "$FREE_MB" -ge "$REQUIRED_FREE_MB" ] 2>/dev/null || fail \
    "Not enough free disk space." \
    "MARP needs about $((REQUIRED_FREE_MB / 1000)) GB free and this disk has $((FREE_MB / 1000)) GB." \
    "Free some space and run this installer again."
say "    disk: $((FREE_MB / 1000)) GB free"

if [ "$CHECK_ONLY" = yes ]; then
    say ""
    say "This computer can run MARP."
    say "  Run this installer without --check-only to set it up."
    exit 0
fi

say "    ready"

# ---------------------------------------------------------------------------
stage "Downloading the verified setup tools..."
# ---------------------------------------------------------------------------
fetch uv-linux-x64.lock.json "$DOWNLOADS/uv.tar.gz"
rm -rf "$INSTALL_ROOT/uv" && mkdir -p "$INSTALL_ROOT/uv"
tar -xzf "$DOWNLOADS/uv.tar.gz" -C "$INSTALL_ROOT/uv" --strip-components=1
UV="$INSTALL_ROOT/uv/uv"
[ -x "$UV" ] || fail "The setup tool did not unpack correctly."

# ---------------------------------------------------------------------------
stage "Installing the managed Python 3.12 runtime..."
# ---------------------------------------------------------------------------
# 3.12 deliberately, matching pyproject.toml's floor and ceiling: torch and
# ultralytics wheels lag new Python releases. A volunteer's system Python is not
# used at all, so a distribution shipping 3.13 or 3.14 does not matter.
UV_PYTHON_INSTALL_DIR="$INSTALL_ROOT/python"; export UV_PYTHON_INSTALL_DIR
"$UV" python install 3.12 >> "$LOG" 2>&1 || fail "Could not install the Python runtime."

# ---------------------------------------------------------------------------
stage "Creating an isolated MARP environment..."
# ---------------------------------------------------------------------------
# Reuse an existing environment rather than recreating it. `uv venv` refuses a
# directory that already holds one, and its suggested `--clear` would delete
# several gigabytes of already-installed packages -- so a volunteer whose first
# attempt died during the long download would pay for the whole thing again.
# Re-running this installer has to be cheap or nobody re-runs it.
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c '' 2>/dev/null; then
    say "    reusing the existing environment"
else
    rm -rf "$VENV"
    "$UV" venv --python 3.12 "$VENV" >> "$LOG" 2>&1 || fail \
        "Could not create the MARP environment." \
        "See the log for what the environment tool reported."
fi

# ---------------------------------------------------------------------------
stage "Installing the MARP worker..."
# ---------------------------------------------------------------------------
say "    this is the long part -- several GB, and it only happens once"
SOURCE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/worker"
# Pin to the checked-in lock when one exists for this variant, so every worker on
# this runtime resolves identically. A variant without a lock resolves freshly --
# worse, and still better than refusing to install.
CONSTRAINT=""
[ -f "$LOCKS/requirements-linux-$TORCH_BACKEND.lock.txt" ] \
    && CONSTRAINT="--constraint $LOCKS/requirements-linux-$TORCH_BACKEND.lock.txt"
[ -d "$SOURCE_DIR" ] || fail "This installer package is missing the worker source."

"$UV" pip install --python "$VENV/bin/python" \
    --extra-index-url "$TORCH_INDEX" \
    --index-strategy unsafe-best-match \
    $CONSTRAINT \
    "$SOURCE_DIR" >> "$LOG" 2>&1 || fail \
    "Could not install the MARP worker." \
    "This is usually a network problem partway through a large download." \
    "Running the installer again will resume rather than start over."

# Prove the GPU actually works before telling anybody it does. An install that
# reports success and then cannot see the card is the worst outcome, because the
# volunteer has no way to tell whether they are contributing.
if [ "$TORCH_BACKEND" = cpu ]; then
    "$VENV/bin/python" -c "
import torch
print('    torch', torch.__version__, '/ running on the processor')
" || fail "MARP installed but could not start." "See the log for what it reported."
else
"$VENV/bin/python" -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('torch installed but no CUDA device is visible')
print('    torch', torch.__version__, '/', torch.cuda.get_device_name(0))
" || fail \
    "MARP installed but could not use the graphics card." \
    "The driver may be older than the CUDA runtime MARP needs." \
    "Update the NVIDIA driver, restart, and run this installer again."
fi

# ---------------------------------------------------------------------------
stage "Installing the video display..."
# ---------------------------------------------------------------------------
# Chromium is bundled rather than borrowed so a volunteer needs no browser, and so
# the packaged build and a development machine take the same path. _find_chromium()
# looks here before anything on the system.
if [ ! -x "$INSTALL_ROOT/chromium/chrome" ]; then
    fetch chromium-linux-x64.lock.json "$DOWNLOADS/chromium.zip"
    rm -rf "$INSTALL_ROOT/chromium"
    unzip -q "$DOWNLOADS/chromium.zip" -d "$INSTALL_ROOT/tmp-chromium"
    mv "$INSTALL_ROOT/tmp-chromium/chrome-linux" "$INSTALL_ROOT/chromium"
    rmdir "$INSTALL_ROOT/tmp-chromium" 2>/dev/null || true
    chmod +x "$INSTALL_ROOT/chromium/chrome" 2>/dev/null || true
fi
[ -x "$INSTALL_ROOT/chromium/chrome" ] || fail "The video display did not unpack correctly."
say "    video display ready"

# ---------------------------------------------------------------------------
stage "Connecting this computer to MARP..."
# ---------------------------------------------------------------------------
# MARP_WORKER_STATE_DIR is set explicitly here AND in the service below, because
# activation and the job loop have different defaults. Getting this wrong spends the
# enrolment and then reports "this worker is not activated", which reads as a
# credential fault and is a path fault. MARP_API#208 records what it cost.
MARP_WORKER_STATE_DIR="$STATE_DIR"; export MARP_WORKER_STATE_DIR
MARP_COORDINATOR_URL="$COORDINATOR_URL"; export MARP_COORDINATOR_URL

if [ -f "$STATE_DIR/worker-credential.dpapi" ]; then
    say "    this computer is already connected to MARP"
else
    MARP_ACTIVATION_CODE="$ENROLMENT_CODE"; export MARP_ACTIVATION_CODE
    "$VENV/bin/marp-worker-activate" >> "$LOG" 2>&1 || fail \
        "Could not connect this computer to MARP." \
        "The coordinator may be unreachable, or this installer's enrolment code may" \
        "have been revoked. Check your internet connection and try again; if it keeps" \
        "failing, the installer needs updating."
    say "    connected"
fi

# ---------------------------------------------------------------------------
stage "Starting MARP..."
# ---------------------------------------------------------------------------
# A user service rather than a boot service: the watch window needs a graphical
# session, and systemd lingering would start the worker with no display at all --
# so the window would silently never open, which on this platform is the failure
# that looks like success.
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/marp-worker.service" <<UNIT
[Unit]
Description=MARP inference worker
After=graphical-session.target
PartOf=graphical-session.target
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory=$INSTALL_ROOT
Environment=MARP_COORDINATOR_URL=$COORDINATOR_URL
Environment=MARP_WORKER_STATE_DIR=$STATE_DIR
Environment=MARP_CHROMIUM_PATH=$INSTALL_ROOT/chromium/chrome
ExecStart=$VENV/bin/marp-worker --screen window
Restart=always
RestartSec=15

[Install]
WantedBy=graphical-session.target
UNIT

if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
    systemctl --user daemon-reload
    systemctl --user enable --now marp-worker.service >> "$LOG" 2>&1 || true
    say "    MARP will now start automatically when you log in"
else
    # A machine without a systemd user session is unusual but not broken. Say how to
    # start it by hand rather than failing an otherwise complete install.
    say "    this system does not use systemd; start MARP with:"
    say "      $VENV/bin/marp-worker --screen window"
fi

say ""
say "Done. This computer is now a MARP volunteer worker."
say ""
say "  It runs in the background and shows a window while it is working."
say "  Stop it:    systemctl --user stop marp-worker"
say "  Start it:   systemctl --user start marp-worker"
say "  Remove it:  systemctl --user disable --now marp-worker; rm -rf $INSTALL_ROOT"
say ""
