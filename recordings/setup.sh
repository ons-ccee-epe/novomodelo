#!/usr/bin/env bash
#
# setup.sh — install the VHS toolchain used to (re)generate the terminal
# recordings in this directory.
#
# Rendering a recording needs three tools:
#   - vhs    : the recorder that reads a .tape script (installed here as a
#              pinned release binary into BIN_DIR — no root required)
#   - ttyd   : the headless terminal vhs drives (>= 1.7.2)
#   - ffmpeg : the encoder that turns captured frames into a GIF
# plus the `novomodelo` binary on PATH — the subject of every recording.
#
# ttyd + ffmpeg come from the system package manager (dnf / apt / brew); vhs is
# downloaded into BIN_DIR and that directory is ensured on PATH. Versions are
# pinned and overridable via the environment.
#
# Usage:  ./recordings/setup.sh
# Env:    BIN_DIR (default ~/.local/bin), VHS_VERSION, TTYD_VERSION
set -euo pipefail

VHS_VERSION="${VHS_VERSION:-0.11.0}"
TTYD_VERSION="${TTYD_VERSION:-1.7.7}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"

log() { printf '\033[1;33m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

TMP=""
cleanup() { [ -n "$TMP" ] && rm -rf "$TMP"; }
trap cleanup EXIT

# --- privilege + package-manager detection ----------------------------------

SUDO=""
if [ "$(id -u)" -ne 0 ] && have sudo; then
  SUDO="sudo"
fi

pkg_install() {
  if have dnf; then
    $SUDO dnf install -y "$@"
  elif have apt-get; then
    $SUDO apt-get update -qq && $SUDO apt-get install -y "$@"
  elif have brew; then
    brew install "$@"
  else
    die "no supported package manager (dnf/apt-get/brew) — install $* manually"
  fi
}

# --- ffmpeg ------------------------------------------------------------------

ensure_ffmpeg() {
  if have ffmpeg; then
    log "ffmpeg present"
    return
  fi
  log "installing ffmpeg"
  # Fedora ships ffmpeg-free in the main repo; it encodes GIF fine and avoids
  # the rpmfusion dance.
  if have dnf; then
    $SUDO dnf install -y ffmpeg-free || pkg_install ffmpeg
  else
    pkg_install ffmpeg
  fi
}

# --- ttyd (package manager first, pinned static binary as fallback) ---------

ensure_ttyd() {
  if have ttyd; then
    log "ttyd present"
    return
  fi
  log "installing ttyd"
  if pkg_install ttyd 2>/dev/null && have ttyd; then
    return
  fi
  [ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ] ||
    die "could not install ttyd via the package manager — install it manually (https://github.com/tsl0922/ttyd)"
  log "package manager lacked ttyd — downloading static binary $TTYD_VERSION"
  mkdir -p "$BIN_DIR"
  curl -fsSL "https://github.com/tsl0922/ttyd/releases/download/${TTYD_VERSION}/ttyd.x86_64" -o "$BIN_DIR/ttyd"
  chmod 0755 "$BIN_DIR/ttyd"
  log "installed ttyd -> $BIN_DIR/ttyd"
}

# --- vhs (pinned release binary) --------------------------------------------

vhs_arch() {
  case "$(uname -m)" in
    x86_64 | amd64) echo "x86_64" ;;
    aarch64 | arm64) echo "arm64" ;;
    *) die "unsupported architecture $(uname -m) — install vhs manually from https://github.com/charmbracelet/vhs/releases" ;;
  esac
}

ensure_vhs() {
  if have vhs && vhs --version 2>/dev/null | grep -qF "$VHS_VERSION"; then
    log "vhs $VHS_VERSION present"
    return
  fi
  local tarball url bin
  tarball="vhs_${VHS_VERSION}_Linux_$(vhs_arch).tar.gz"
  url="https://github.com/charmbracelet/vhs/releases/download/v${VHS_VERSION}/${tarball}"
  TMP="$(mktemp -d)"
  log "downloading vhs $VHS_VERSION"
  curl -fsSL "$url" -o "$TMP/$tarball" || die "download failed: $url"
  tar -xzf "$TMP/$tarball" -C "$TMP"
  bin="$(find "$TMP" -type f -name vhs | head -1)"
  [ -n "$bin" ] || die "vhs binary not found inside $tarball"
  mkdir -p "$BIN_DIR"
  install -m 0755 "$bin" "$BIN_DIR/vhs"
  log "installed vhs -> $BIN_DIR/vhs"
}

# --- PATH --------------------------------------------------------------------

ensure_path() {
  case ":$PATH:" in
    *":$BIN_DIR:"*) return ;;
  esac
  local rc
  case "${SHELL:-}" in
    *zsh) rc="$HOME/.zshrc" ;;
    *) rc="$HOME/.bashrc" ;;
  esac
  local line="export PATH=\"$BIN_DIR:\$PATH\""
  if ! grep -qsF "$line" "$rc"; then
    printf '\n# added by novomodelo recordings/setup.sh\n%s\n' "$line" >>"$rc"
    log "added $BIN_DIR to PATH in $rc"
  fi
  log "for this shell, run:  export PATH=\"$BIN_DIR:\$PATH\""
}

# --- run ---------------------------------------------------------------------

ensure_ffmpeg
ensure_ttyd
ensure_vhs
ensure_path

if ! have novomodelo; then
  log "note: 'novomodelo' is not on PATH — install it before recording:"
  log "      cargo install --path crates/novomodelo-cli"
fi

log "done. next: ./recordings/generate.sh"
