# gpuq

A GPU queue for a few personal machines shared by one person, many AI agent sessions and a couple of always-on
inference servers. Every card on every machine has its own numbered line. Jobs take turns and give the card up
**cooperatively**: a running job is asked to yield through its heartbeat, stops itself, and resumes when it is first
in line again. `gpuq where` tells you which card on which machine can take a job, and how soon.

- **Fleet-wide placement, no broker:** each host keeps its own queue in JSON files; other hosts are reached over SSH.
- **NVIDIA and AMD, discrete and integrated:** `nvidia-smi` for NVIDIA, sysfs for AMD (no amd-smi), an iGPU sized by
  its shared memory.
- **A declared inventory:** cap, reserve and tag each card's memory, and declare the resident servers that live on it
  (presets for Ollama, vLLM, llama.cpp `llama-server`, llama-swap). `gpuq free` unloads their models for a big job.
- **Agents and humans share one line:** a CLI with exit codes and `--json`, an MCP server, and a terminal UI where
  you reorder the line by hand.
- **Short jobs first, from real numbers:** without `--est`, a job's estimate is the median of its past runs.

Stdlib-only Python (3.9+), Linux. gpuq is advisory: it does not kill your processes or meter memory use. It keeps
an honest, visible line and flags GPU use that did not go through it.

## Install

    pip install .          # or: pipx install .   (provides `gpuq` and `gpuq-mcp`)
    # no install: PYTHONPATH=. python3 -m gpuq
    gpuq init              # write a starter config from the cards (and servers) found here

State lives in `~/.local/state/gpuq` (override: `$GPUQ_STATE`). Run `gpuq sweep` every minute from a timer
(systemd user timer or cron); it purges dead holders.

### An app-menu launcher

`examples/gpuq.desktop` puts "GPU Queue" in your app menu, like htop's entry, so the live view opens in its own
window and closes without a "running program" prompt. Copy it to `~/.local/share/applications/`. It uses your
default terminal at its default size.
For a bigger window (130x40 fits the full layout with bars and the model column), set `Terminal=false` and start
the terminal yourself on the `Exec=` line:

| terminal | `Exec=` |
|---|---|
| GNOME Terminal | `gnome-terminal --geometry=130x40 -- gpuq` |
| Konsole | `konsole -p TerminalColumns=130 -p TerminalRows=40 -e gpuq` |
| kitty | `kitty -o initial_window_width=130c -o initial_window_height=40c gpuq` |
| xterm | `xterm -geometry 130x40 -e gpuq` |

Some terminals (Ptyxis, for one) can't set a window size per launch, only a default size for every window;
use another terminal for the launcher, or change that default. Use the full path to `gpuq` if the menu's PATH
lacks `~/.local/bin`.

## Quickstart

    gpuq where --vram 16                 # which card, on which machine, how soon
    ID=$(gpuq join --owner nightly --job "eval sweep" --vram 16 --est 90 --max 180)
    gpuq run $ID -- python eval.py       # waits its turn, runs, heartbeats, releases
    gpuq                                 # live view (like htop): k/j move jobs, Tab switches GPU, q quits
    gpuq show                            # print the line once (what `gpuq` prints when piped)
    gpuq inventory                       # what every card offers

## The agent protocol

| call | meaning |
|---|---|
| `gpuq join ...` | get in line, prints an id. `--vram GB` / `--tag T` pick a card that fits (shortest wait, then the smallest card); `--gpu` names one. `--est` minutes (default: the median of this job's past runs). `--priority chill\|normal\|important\|ablaze\|preempt\|N` (below) `--spec FILE` (or `-` for stdin) reads all of it as one JSON object |
| `gpuq start ID` | exit 0 = you hold the card. Exit **3** = not your turn yet (or a yield was acknowledged): retry in a minute. Prints `CUDA_VISIBLE_DEVICES=GPU-<uuid>` (or `ROCR_VISIBLE_DEVICES=…`) on multi-card hosts |
| `gpuq beat ID [--progress 41/80]` | heartbeat, about every 60 s (silent 15 min = purged). Exit **4** = you were purged: stop. Exit **5** = yield: stop your GPU work, go back to polling `start` |
| `gpuq done ID` | finished, leave the line |
| `gpuq run ID [--free] [--on-stop SH] [--progress-file F] -- CMD...` | does all of the above. On a yield it sends CMD's process group SIGTERM (SIGKILL after `--grace`, default 30 s) and **reruns CMD from the start** when you are #1 again, so CMD must be safe to rerun (checkpoint yourself). `--free` frees the card's residents once granted and restarts what it stopped at the end |

Add `--json` to `join`, `start`, `beat`, `done`, `env`, `edit`, `mv`, `yield` or `rm` for one JSON object instead:
`{"ok", "code", "state", "message", …}` where `state` is `go` / `wait` / `gone` for `start` and `ok` / `yield` /
`purged` for `beat`. `start --json` carries the pin as `"env": {"CUDA_VISIBLE_DEVICES": "GPU-…"}`.

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

Priority: any integer, higher goes first; the names are defaults: `chill` 5 (☾), `normal` 10, `important` 20 (★),
`ablaze` 30 (⚑), and `preempt` = one above the running job's. A joining job goes just behind the last job of at least
its priority, and outranking the running job asks it to yield (it resumes after). `--chill`, `--important`, `--ablaze`
and `--preempt` are shortcuts; `gpuq edit ID --priority …` re-places a job.

Rules of the line: a yield is asked by moving the running job down (`gpuq mv 1 3`, TUI `j`, or `gpuq yield ID`).
The yielder keeps the card until its next `start` acknowledges. A waiting #1 that does not `start` within 15 min is
skipped to the back (twice: removed). A job of at most 6 h goes ahead of jobs with 10x more time left (decided at
`join`/`edit` only, never otherwise) unless they are important (priority 20+) or outrank it. A pause (`gpuq pause`) lets nobody start.

## MCP

`gpuq mcp` (or `gpuq-mcp`) is an MCP server over stdio, stdlib only. Point your agent's MCP config at it:

    {"mcpServers": {"gpuq": {"command": "gpuq-mcp"}}}

Tools: `gpuq_inventory`, `gpuq_where`, `gpuq_show`, `gpuq_join`, `gpuq_start`, `gpuq_beat`, `gpuq_done`; each takes
an optional `host`. They return the same states as `--json`, plus a `next` hint for the agent.

Agents forget to heartbeat, so any tool call also beats every job that MCP session has started, and a pending yield
for one of them comes back on *any* result as a `notices` entry. A session that makes no calls at all still goes
stale after 15 minutes. For long work, `gpuq run ID -- CMD` in a shell beats by itself: use MCP to ask and hold,
`gpuq run` to run.

## Inventory and placement

Cards are detected; the config says what they offer. Usable memory = the card's memory (capped by `vram_gb`) minus
`reserve_gb` minus what its residents hold. `gpuq inventory` shows it per card, with tags, residents, the line and
the wait. `gpuq where --vram GB [--tag T] [--est MIN]` ranks every card on every host: cards that fit first, then the
shortest wait (with the short-job jump applied when `--est` is given), then this host, then the smallest card. It
exits 1 when nothing fits.

Tags: `cuda` or `rocm` by vendor, `igpu` for integrated GPUs, plus any you add per card.

Residents are declared per card, either as a process name (the old form) or as an object:

    "residents": {
      "3090": [{"type": "ollama", "vram_gb": 8}],
      "3060": ["whisper-server", {"type": "vllm", "unit": "vllm-small.service"}]
    }

| field | meaning |
|---|---|
| `type` | a preset: `ollama`, `vllm`, `llama-server`, `llama-swap` (fills in the fields below) |
| `name` | label (default: the type or process name) |
| `match` | process name as gpuq sees it, to recognize it on a card |
| `unit` | its systemd user unit (`evac` restarts it pinned elsewhere; `free` may stop it) |
| `vram_gb` | memory it normally holds; `vram_frac` = a share of the card (the vLLM preset assumes 0.9) |
| `free` | how `gpuq free` releases it: `ollama` / `llama-swap` (unload over HTTP at `url`), `stop` (stop the unit; `restore` starts it), or a shell command |

`gpuq free [CARD]` unloads or stops the card's residents (`--dry-run` previews). Ollama and llama-swap load models
again on the next request, so nothing needs restoring; stopped units come back with `gpuq restore`. A resident with no
`type` or `free` is reported and left alone. `gpuq evac [CARD]` instead moves the residents to another card with a
systemd drop-in and a restart; `gpuq restore` undoes either.

gpuq only plans with what you declare. It does not read the servers' own configs, guess which model they will
load, or decide when a resident may be evicted.

## Config

Optional JSON at `~/.config/gpuq/config.json` (`$GPUQ_CONFIG`); `gpuq init` writes a starter one, and
`examples/config.json` shows every key.

| key | effect |
|---|---|
| `default` | card (by model number, e.g. `3090`) used when a job names none; else the card with most memory |
| `owners` | `{owner: card}` default card per owner |
| `cards` | `{card: {"reserve_gb", "vram_gb", "tags"}}` per card |
| `residents` | `{card: [process name or resident object]}`, above |
| `units` | `{process name: systemd user unit}` for old-form residents (default `<name>.service`) |
| `hosts` | comma list for the multi-host views and `where` (default: this host) |
| `remote_command` | gpuq on other hosts over non-interactive ssh (default `~/.local/bin/gpuq`) |
| `rules` | `{"jump_ratio": 10, "jump_max_hours": 6, "priorities": {"chill": 5, "normal": 10, "important": 20, "ablaze": 30}}` |
| `compat_lock` | a lock file for older tools; written while a job runs, removed after |

## Cards and hosts

Each card has its own line, named by model number. NVIDIA jobs get their card by UUID in `CUDA_VISIBLE_DEVICES`
(nvidia-smi and CUDA number cards differently). Every host keeps its own queue; `gpuq --host H cmd`, `show`, `watch`,
`inventory`, `where` and `tui --hosts a,b` reach others over SSH (key auth, gpuq installed there). `where` suggests
a host; nothing places jobs across hosts for you.

### AMD and integrated GPUs

AMD cards are found in `/sys/class/drm` (driver `amdgpu`) and named from the PCI id database (`Radeon 880M` -> `880M`);
no vendor tool is needed. An integrated GPU has a tiny dedicated carve-out and uses system RAM, so its size is shown as
shared memory (`880M (24 GB shared)`), and a discrete card is preferred as the host default. Cap what an iGPU may lend
with `cards.<name>.vram_gb` (`gpuq init` proposes half). Jobs on an AMD card get `ROCR_VISIBLE_DEVICES` (the ROCm
device index, or `GPU-<unique id>` when the card has one); `gpuq run` sets it, and `gpuq start` / `gpuq env ID` print
it. Only `ROCR_VISIBLE_DEVICES` is set, not `HIP_VISIBLE_DEVICES`, because HIP applies its own filter on top of ROCm's
and the two would compound. "processes on this card" lists processes holding at least 1 GB, read from the kernel's DRM fdinfo
(needs kernel 5.19+ for the memory keys; you only see processes you are allowed to inspect), plus `gpu_busy_percent`.
Not covered: Intel GPUs, Vulkan-only devices without `amdgpu`, and a job that spans several cards. A host that has
neither tool still works as a plain numbered line without pinning or usage warnings. Mixed NVIDIA + AMD hosts get one
line per card of either kind.

## Limits

Memory accounting is declarative: `--vram` is checked against what the card offers after reserves and residents,
not against live usage. Jobs on a card run one at a time. No priorities beyond order and the short-job rule, no
checkpointing, nothing enforces the line against processes that ignore it (they show up as "unqueued GPU use").
Linux only. Purging a holder removes its place; it does not kill its process.

## Tests

    python3 -m unittest discover -s tests

They run against temporary state directories and fake cards; nothing touches a real queue, GPU, systemd unit or
inference server.

## License

MIT, see LICENSE.
