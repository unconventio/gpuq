#!/usr/bin/env bash
# Install the gpuq top-bar indicator (GNOME Shell extension) for the current user. No sudo.
#   ./install.sh --experimental   install (or update) and enable
#   ./install.sh --remove         disable and remove
# The extension is experimental: it only installs when asked for by that flag.
set -euo pipefail
UUID="gpuq@unconventio"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/extension"
DEST="$HOME/.local/share/gnome-shell/extensions/$UUID"
if [ "${1:-}" = --remove ]; then
    gnome-extensions disable "$UUID" 2>/dev/null || true
    rm -rf "$DEST"; echo "removed $DEST"; exit 0
fi
if [ "${1:-}" != --experimental ]; then
    echo "The gpuq top-bar extension is experimental. Install it anyway with: $0 --experimental" >&2
    exit 2
fi
command -v gpuq >/dev/null || [ -x "$HOME/.local/bin/gpuq" ] || echo "warning: gpuq is not on PATH or in ~/.local/bin" >&2
major="$(gnome-shell --version 2>/dev/null | grep -oE '[0-9]+' | head -1 || true)"
if [ -n "$major" ] && ! grep -q "\"$major\"" "$SRC/metadata.json"; then
    echo "warning: GNOME Shell $major is not in metadata.json shell-version; add it first" >&2
fi
mkdir -p "$DEST"; cp "$SRC"/*.js "$SRC"/*.json "$SRC"/*.css "$DEST/"; mkdir -p "$DEST/icons"; cp "$SRC"/icons/*.svg "$DEST/icons/"
gnome-extensions enable "$UUID" 2>/dev/null || {
    cur="$(gsettings get org.gnome.shell enabled-extensions)"
    case "$cur" in
        *"'$UUID'"*) ;;
        "@as []") gsettings set org.gnome.shell enabled-extensions "['$UUID']" ;;
        *) gsettings set org.gnome.shell enabled-extensions "${cur%]}, '$UUID']" ;;
    esac
}
echo "installed to $DEST and enabled."
gnome-extensions info "$UUID" 2>/dev/null | grep -q 'State: ACTIVE' && echo "state: ACTIVE" \
    || echo "On Wayland a new extension loads at the next login: log out and back in once."
