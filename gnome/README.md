# gpuq top-bar indicator (GNOME Shell) — experimental

A GNOME Shell extension that puts gpuq in the top bar: one icon, coloured by state, and a dropdown with
every host's cards, queues and temperatures. Jobs can be moved, and (after a second click) asked to
yield or removed. It only reads `gpuq show --json` and runs gpuq commands; it keeps no state of its own.

Experimental: off by default, and only installed when you ask for it.

    ./install.sh --experimental     # install or update, then log out and back in (Wayland)
    ./install.sh --remove

Tested on GNOME Shell 50. To try it without logging out, run a nested shell:
`dbus-run-session gnome-shell --devkit --wayland` (needs the `mutter-dev-bin` package).
