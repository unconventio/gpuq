# gpuq

A numbered GPU queue for a shared workstation. People and AI agent sessions join a line per card, take turns,
and give the card up **cooperatively**: a running job is asked to yield through its heartbeat, stops itself,
and resumes when it is first in line again. Stdlib-only Python (3.9+), Linux. NVIDIA cards are read from `nvidia-smi`, AMD cards (discrete and integrated) from sysfs.

gpuq is advisory. It does not kill your processes or meter VRAM; it keeps an honest, visible line and flags
GPU use that did not go through it.

## Install

    pip install .          # or: pipx install .   (provides the `gpuq` command)
    # no install: PYTHONPATH=. python3 -m gpuq

State lives in `~/.local/state/gpuq` (override: `$GPUQ_STATE`). Run `gpuq sweep` every minute from a timer
(systemd user timer or cron); it purges dead holders.

## Quickstart

    ID=$(gpuq join --owner nightly --job "eval sweep" --why "weekly regression" --models qwen3:8b --est 90 --max 180)
    gpuq run $ID -- python eval.py       # waits its turn, runs, heartbeats, releases
    gpuq                                 # look at the line
    gpuq tui                             # interactive: move jobs with k/j, d removes, p pauses

## The agent protocol

| call | meaning |
|---|---|
| `gpuq join ...` | get in line, prints an id. Always pass `--est` (minutes); `--important` stops short jobs jumping you |
| `gpuq start ID` | exit 0 = you hold the card. Exit **3** = not your turn yet (or a yield was acknowledged): retry in a minute. Prints `CUDA_VISIBLE_DEVICES=GPU-<uuid>` on multi-card hosts |
| `gpuq beat ID [--progress 41/80]` | heartbeat, about every 60 s (silent 15 min = purged). Exit **4** = you were purged: stop. Exit **5** = yield: stop your GPU work, go back to polling `start` |
| `gpuq done ID` | finished, leave the line |
| `gpuq run ID [--on-stop SH] [--progress-file F] -- CMD...` | does all of the above. On a yield it sends CMD's process group SIGTERM (SIGKILL after `--grace`, default 30 s) and **reruns CMD from the start** when you are #1 again, so CMD must be safe to rerun (checkpoint yourself) |

Hand-rolled holder loop:

    while :; do
      until gpuq start $ID; do [ $? = 3 ] || exit; sleep 60; done     # 4 = purged
      launch the GPU work
      while work is alive; do
        gpuq beat $ID --progress "$P"; rc=$?
        [ $rc = 4 ] && exit
        [ $rc = 5 ] && { stop the work; break; }                      # yield
        sleep 60
      done
      work finished on its own -> gpuq done $ID; exit
    done

Rules of the line: a yield is asked by moving the running job down (`gpuq mv 1 3`, TUI `j`, or `gpuq yield ID`).
The yielder keeps the card until its next `start` acknowledges. A waiting #1 that does not `start` within 15 min is
skipped to the back (twice: removed). A job of at most 6 h goes ahead of jobs with 10x more time left (decided at
`join`/`edit` only, never otherwise) unless they are `--important`. A pause (`gpuq pause`) lets nobody start.

## Config

Optional JSON at `~/.config/gpuq/config.json` (`$GPUQ_CONFIG`); annotated example in `examples/config.json`.

| key | effect |
|---|---|
| `default` | card (by model number, e.g. `3090`) used when a job names none; else the card with most memory |
| `owners` | `{owner: card}` default card per owner |
| `residents` | `{card: [process names]}` always-on services: shown, never "unqueued use", flagged if on the wrong card |
| `units` | `{process name: systemd user unit}` for `evac` (default `<name>.service`) |
| `hosts` | comma list for the multi-host views (default: this host) |
| `remote_command` | gpuq path on other hosts over non-interactive ssh (default `~/.local/bin/gpuq`) |
| `compat_lock` | a lock file for older tools; written while a job runs, removed after |

## Cards, residents, hosts

Each card has its own line, named by model number. NVIDIA jobs get their card by UUID in `CUDA_VISIBLE_DEVICES`
(nvidia-smi and CUDA number cards differently). `gpuq evac [CARD]` moves a card's resident services to another card
with a systemd drop-in and a restart, `gpuq restore` undoes it (`--dry-run` previews both).
Every host keeps its own queue; `gpuq --host H cmd`, `show`, `watch` and `tui --hosts a,b` reach others over SSH
(key auth, gpuq installed there). There is no cross-host placement: you pick the host.

### AMD and integrated GPUs

AMD cards are found in `/sys/class/drm` (driver `amdgpu`) and named from the PCI id database (`Radeon 880M` -> `880M`);
no vendor tool is needed. An integrated GPU has a tiny dedicated carve-out and uses system RAM, so its size is shown as
shared memory (`880M (24 GB shared)`), and a discrete card is preferred as the host default. Jobs on an AMD card get
`ROCR_VISIBLE_DEVICES` (the ROCm device index, or `GPU-<unique id>` when the card has one); `gpuq run` sets it, and
`gpuq start` / `gpuq env ID` print it. Only `ROCR_VISIBLE_DEVICES` is set, not `HIP_VISIBLE_DEVICES`, because HIP applies
its own filter on top of ROCm's and the two would compound. "GPU in use" lists processes holding at least 1 GB, read from
the kernel's DRM fdinfo (needs kernel 5.19+ for the memory keys; you only see processes you are allowed to inspect),
plus `gpu_busy_percent`. Not covered: Intel GPUs, Vulkan-only devices without `amdgpu`, and a job that spans several cards.
A host that has neither tool still works as a plain numbered line without pinning or usage warnings.
Mixed NVIDIA + AMD hosts get one line per card of either kind.

## Limits

No VRAM accounting, no priorities beyond order and the short-job rule, no checkpointing, nothing enforces the line
against processes that ignore it (they show up as "unqueued GPU use"). Linux only. Purging a holder
removes its place; it does not kill its process.

## Tests

    python3 -m unittest discover -s tests

They run against a temporary state directory.

License: MIT, see LICENSE.
