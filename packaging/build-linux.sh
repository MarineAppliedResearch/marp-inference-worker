#!/bin/sh
# build-linux.sh
# Created: 2026-09-18
# Author: Isaac Travers
#
# Builds the thing a volunteer downloads: one self-extracting shell script that
# installs MARP on a Linux computer.
#
# The POSIX counterpart of build-windows.ps1, and the same division of labour --
# everything that needs a decision, a credential or a clean tree happens here on a
# build machine, and the artifact a volunteer runs makes none of them.
#
# The enrolment code is a *build input*. `bootstrap-linux.sh` carries the
# placeholder `@MARP_ENROLMENT_CODE@` in the repository and the real value only
# ever exists in the built artifact. MARP_API#208 makes that a requirement:
# "The code is never committed to either repository."
#
# Output is one file because that is the whole promise. A volunteer downloads it,
# runs it, and is done -- no unpacking, no reading a README, no second step.

set -eu

HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO="$(CDPATH= cd -- "$HERE/.." && pwd)"
OUT_DIR="${MARP_BUILD_OUT:-$REPO/dist}"

ENROLLMENT_CODE=""
COORDINATOR_URL=""
ALLOW_DIRTY=no


usage() {
    printf '%s\n' \
        "Build the MARP Linux volunteer installer." \
        "" \
        "  --enrollment-code CODE   the standing enrolment code (required)" \
        "  --coordinator-url URL    the coordinator (required)" \
        "  --allow-dirty            build from an unclean tree, for testing only" \
        "" \
        "The parameter name matches build-windows.ps1's -EnrollmentCode so the two" \
        "builds differ in mechanism rather than in interface." \
        "" \
        "The code can also come from MARP_ENROLLMENT_CODE in the environment, which" \
        "is what CI should do -- an argument is visible in the process list to every" \
        "user on the build machine."
}


while [ $# -gt 0 ]; do
    case "$1" in
        --enrollment-code) ENROLLMENT_CODE="$2"; shift 2 ;;
        --coordinator-url) COORDINATOR_URL="$2"; shift 2 ;;
        --allow-dirty)     ALLOW_DIRTY=yes; shift ;;
        --help|-h)         usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

# Environment beats argument, because an argument is readable by every user on the
# machine through /proc while the build runs.
ENROLLMENT_CODE="${MARP_ENROLLMENT_CODE:-$ENROLLMENT_CODE}"
COORDINATOR_URL="${MARP_COORDINATOR_URL:-$COORDINATOR_URL}"

[ -n "$ENROLLMENT_CODE" ] || { printf 'An enrolment code is required.\n\n' >&2; usage >&2; exit 2; }
[ -n "$COORDINATOR_URL" ] || { printf 'A coordinator URL is required.\n\n' >&2; usage >&2; exit 2; }

# A build from an unclean tree produces an artifact nobody can reproduce, and the
# one thing worse than no installer is one whose contents cannot be identified
# after a volunteer reports a problem with it.
if [ "$ALLOW_DIRTY" = no ]; then
    [ -z "$(git -C "$REPO" status --porcelain)" ] || {
        printf 'The working tree has uncommitted changes.\n' >&2
        printf 'Commit them, or pass --allow-dirty for a throwaway test build.\n' >&2
        exit 1
    }
fi

REVISION=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT INT TERM

printf 'Building MARP Linux installer from %s\n' "$REVISION"

# ---------------------------------------------------------------------------
# Stage the payload: the bootstrap, its locks, and the worker source.
# ---------------------------------------------------------------------------
PAYLOAD="$STAGE/payload"
mkdir -p "$PAYLOAD/worker"

cp "$HERE/bootstrap-linux.sh" "$PAYLOAD/"
cp "$HERE"/chromium-linux-x64.lock.json "$HERE"/uv-linux-x64.lock.json "$PAYLOAD/"
# Every requirements lock, because the variant is chosen on the volunteer's machine
# from their card rather than here.
cp "$HERE"/requirements-linux-*.lock.txt "$PAYLOAD/" 2>/dev/null || true

cp -r "$REPO/src" "$REPO/pyproject.toml" "$PAYLOAD/worker/"
cp -r "$REPO/ByteTrack" "$PAYLOAD/worker/"

# ByteTrack's demo GIFs and sample video are roughly 62 MB of documentation that no
# volunteer runs. The registry already notes it. Dropped here rather than from the
# vendored tree, which stays byte-identical to upstream on purpose.
rm -rf "$PAYLOAD/worker/ByteTrack/assets" "$PAYLOAD/worker/ByteTrack/videos"
# Build artifacts from a developer's tree must not travel: an editable install
# leaves an egg-info whose paths are this machine's.
find "$PAYLOAD/worker" -name '*.egg-info' -prune -exec rm -rf {} + 2>/dev/null || true
find "$PAYLOAD/worker" -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Substitute the build inputs. This is the only place the real code exists.
# ---------------------------------------------------------------------------
# A literal `sed s|...|CODE|` would put the code in the process list, so it is fed
# through the environment and substituted by a tool that reads it from there.
MARP_SUBST_CODE="$ENROLLMENT_CODE" MARP_SUBST_URL="$COORDINATOR_URL" \
python3 - "$PAYLOAD/bootstrap-linux.sh" <<'SUBST'
import os, sys, pathlib
p = pathlib.Path(sys.argv[1])
text = p.read_text()
for placeholder, value in (
    ("@MARP_ENROLMENT_CODE@", os.environ["MARP_SUBST_CODE"]),
    ("@MARP_COORDINATOR_URL@", os.environ["MARP_SUBST_URL"]),
):
    if placeholder not in text:
        sys.exit(f"{placeholder} is missing from bootstrap-linux.sh; the build cannot "
                 f"produce a working installer without it")
    text = text.replace(placeholder, value)
p.write_text(text)
SUBST

# ---------------------------------------------------------------------------
# Wrap it as one self-extracting script.
# ---------------------------------------------------------------------------
mkdir -p "$OUT_DIR"
ARTIFACT="$OUT_DIR/marp-worker-setup-linux-x86_64.sh"

tar -czf "$STAGE/payload.tar.gz" -C "$PAYLOAD" .

# The header is plain POSIX sh and the archive is appended after a marker line, so
# there is no base64 and the artifact stays the size of its contents. `awk` finds
# the marker rather than a hard-coded line count, which would silently break the
# moment anybody edits a line of the header.
cat > "$ARTIFACT" <<'HEADER'
#!/bin/sh
# MARP volunteer worker setup.
#
# One file. Run it, and this computer joins MARP.
#   sh marp-worker-setup-linux-x86_64.sh
#
# To check the computer is suitable without installing anything:
#   sh marp-worker-setup-linux-x86_64.sh --check-only
set -eu

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT INT TERM

# Everything below the marker is a gzipped tar of the installer and its payload.
ARCHIVE_LINE=$(awk '/^__MARP_PAYLOAD__$/ { print NR + 1; exit 0; }' "$0")
tail -n "+$ARCHIVE_LINE" "$0" | tar -xzf - -C "$WORK"

sh "$WORK/bootstrap-linux.sh" "$@"
exit $?
__MARP_PAYLOAD__
HEADER

cat "$STAGE/payload.tar.gz" >> "$ARTIFACT"
chmod +x "$ARTIFACT"

SIZE_MB=$(du -m "$ARTIFACT" | cut -f1)
printf '\n'
printf 'Built %s\n' "$ARTIFACT"
printf '  revision %s, %s MB\n' "$REVISION" "$SIZE_MB"
printf '\n'
printf '  This artifact contains the enrolment code. Anyone holding it can enrol a\n'
printf '  worker, so publish it where you intend volunteers to find it and nowhere else.\n'
printf '\n'
