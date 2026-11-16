// gpuq — GNOME Shell top-bar view of the gpuq GPU queue. EXPERIMENTAL (install.sh --experimental).
//
// Bar: one icon (a card whose fan is a Q). Amber = GPU use outside the queue or a hot card; grey = this
// host's queue is paused or unreadable. Menu: a one-line summary ("3090▶+2 3060· laptop?"), then each
// host's cards (memory, load, temperature) and their lines (position, owner, job, progress, est/max, how
// long running or waiting); each job opens a submenu to move it, or (with a second click) to yield or
// remove it. Residents and unqueued processes are listed per card.
// Data: `gpuq show --json --local` here every 12 s; other hosts in the gpuq config ("hosts") with
// `gpuq --host H show --json` every 60 s. Every call is an async subprocess with a kill timer, so a
// slow or unreachable host never blocks the shell.

import GLib from 'gi://GLib';
import Gio from 'gi://Gio';
import St from 'gi://St';
import Clutter from 'gi://Clutter';
import GObject from 'gi://GObject';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js';
import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import {cards, chunks, entryRow, cardTitle, fmtDur} from './lib.js';

const HOME = GLib.get_home_dir();
const CONFIG = GLib.build_filenamev([HOME, '.config', 'gpuq', 'config.json']);
const LOCAL_EVERY_S = 12;
const REMOTE_EVERY_S = 60;
const LOCAL_TIMEOUT_S = 8;
const REMOTE_TIMEOUT_S = 10;   // ssh ConnectTimeout is 5 s; this also covers a host that connects but hangs

function gpuqPath() {
    return GLib.find_program_in_path('gpuq') ?? GLib.build_filenamev([HOME, '.local', 'bin', 'gpuq']);
}

// Run argv asynchronously; resolve to stdout text, or null on failure or timeout. Never throws.
function runAsync(argv, timeoutS) {
    return new Promise(resolve => {
        let proc, timer = 0, done = false;
        const finish = out => {
            if (done)
                return;
            done = true;
            if (timer)
                GLib.source_remove(timer);
            resolve(out);
        };
        try {
            proc = Gio.Subprocess.new(argv, Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_SILENCE);
        } catch (e) {
            finish(null);
            return;
        }
        timer = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, timeoutS, () => {
            timer = 0;
            proc.force_exit();
            finish(null);
            return GLib.SOURCE_REMOVE;
        });
        proc.communicate_utf8_async(null, null, (p, res) => {
            try {
                const [, out] = p.communicate_utf8_finish(res);
                finish(p.get_successful() ? out : null);
            } catch (e) {
                finish(null);
            }
        });
    });
}

function readFileAsync(path) {
    return new Promise(resolve => {
        Gio.File.new_for_path(path).load_contents_async(null, (f, res) => {
            try {
                const [, bytes] = f.load_contents_finish(res);
                resolve(new TextDecoder().decode(bytes));
            } catch (e) {
                resolve(null);
            }
        });
    });
}

function terminalArgv(cmd) {
    for (const [prog, ...pre] of [['ptyxis', '--new-window', '--'], ['gnome-terminal', '--'], ['kgx', '--'],
        ['konsole', '-e'], ['xfce4-terminal', '-x'], ['x-terminal-emulator', '-e']]) {
        const p = GLib.find_program_in_path(prog);
        if (p)
            return [p, ...pre, ...cmd];
    }
    return null;
}

// Run a gpuq action; resolve to {ok, out} with stdout+stderr. Never throws.
function actAsync(argv, timeoutS) {
    return new Promise(resolve => {
        let proc, timer = 0, done = false;
        const finish = r => {
            if (done)
                return;
            done = true;
            if (timer)
                GLib.source_remove(timer);
            resolve(r);
        };
        try {
            proc = Gio.Subprocess.new(argv, Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_MERGE);
        } catch (e) {
            finish({ok: false, out: e.message});
            return;
        }
        timer = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, timeoutS, () => {
            timer = 0;
            proc.force_exit();
            finish({ok: false, out: `timed out after ${timeoutS} s`});
            return GLib.SOURCE_REMOVE;
        });
        proc.communicate_utf8_async(null, null, (p, res) => {
            try {
                const [, out] = p.communicate_utf8_finish(res);
                finish({ok: p.get_successful(), out: (out ?? '').trim()});
            } catch (e) {
                finish({ok: false, out: e.message});
            }
        });
    });
}

// A menu item that needs a second click: the first one only re-labels it and keeps the menu open.
const ConfirmItem = GObject.registerClass(
class ConfirmItem extends PopupMenu.PopupMenuItem {
    _init(text, onArm) {
        super._init(text);
        this._text = text;
        this._onArm = onArm;
        this._armed = false;
    }

    activate(event) {
        if (!this._armed) {
            this._armed = true;
            this.label.text = `Click again to confirm: ${this._text}`;
            this._onArm?.();
            return;
        }
        super.activate(event);
    }
});

const Indicator = GObject.registerClass(
class Indicator extends PanelMenu.Button {
    _init(dir) {
        super._init(0.0, 'gpuq', false);
        this._local = GLib.get_host_name().split('.')[0];
        this._hosts = [this._local];
        this._views = new Map();       // host -> {host, local, snap, err, at, busy}
        // The bar is just the icon (a card with a Q); its colour is the state, the menu has the rest.
        this._icon = new St.Icon({
            gicon: Gio.icon_new_for_string(GLib.build_filenamev([dir, 'icons', 'gpuq-symbolic.svg'])),
            style_class: 'system-status-icon gpuq-icon gpuq-unknown',
        });
        this.add_child(this._icon);
        this._summary = '';
        this.menu.connect('open-state-changed', (m, open) => {
            if (open)
                this._poll(true);
            else if (this._stale)
                this._rebuildMenu();
        });
        this._rebuildMenu();
        this._loadHosts().then(() => this._poll(true));
        this._timer = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, LOCAL_EVERY_S, () => {
            this._poll(false);
            return GLib.SOURCE_CONTINUE;
        });
    }

    async _loadHosts() {
        const text = await readFileAsync(CONFIG);
        let hosts = null;
        try {
            hosts = JSON.parse(text ?? '{}').hosts;
        } catch (e) {}
        const list = (typeof hosts === 'string' ? hosts.split(',') : []).map(h => h.trim()).filter(Boolean);
        this._hosts = [this._local, ...list.filter(h => h !== this._local && h !== 'localhost')];
    }

    _poll(force) {
        const now = GLib.get_monotonic_time() / 1e6;
        for (const host of this._hosts) {
            const local = host === this._local;
            const v = this._views.get(host) ?? {host, local, snap: null, at: -Infinity, busy: false};
            this._views.set(host, v);
            const every = local ? LOCAL_EVERY_S - 1 : force ? 15 : REMOTE_EVERY_S;
            if (v.busy || now - v.at < every)
                continue;
            v.busy = true;
            const argv = local ? [gpuqPath(), 'show', '--json', '--local'] : [gpuqPath(), '--host', host, 'show', '--json'];
            runAsync(argv, local ? LOCAL_TIMEOUT_S : REMOTE_TIMEOUT_S).then(out => {
                v.busy = false;
                v.at = GLib.get_monotonic_time() / 1e6;
                v.wall = Date.now() / 1000;
                let snap = null;
                try {
                    snap = out ? JSON.parse(out) : null;
                } catch (e) {}
                v.snap = snap;      // unreachable or broken: shown as "host ?" rather than stale data
                this._render();
            });
        }
    }

    _ordered() {
        return this._hosts.map(h => this._views.get(h)).filter(Boolean);
    }

    // Icon colour: amber if any card has unqueued GPU use or runs hot; grey if this host's queue is paused or unreadable.
    _setState(list) {
        const local = this._views.get(this._local);
        const cls = list.some(c => c.cls === 'gpuq-unqueued') ? 'gpuq-unqueued'
            : !local?.snap || local.snap.paused ? 'gpuq-paused' : '';
        for (const c of ['gpuq-unqueued', 'gpuq-paused', 'gpuq-unknown'])
            this._icon.remove_style_class_name(c);
        if (cls)
            this._icon.add_style_class_name(cls);
        this._summary = list.map(c => c.text).join('  ');
        this.accessible_name = `gpuq ${this._summary}`;
    }

    _render() {
        if (this._destroyed)
            return;
        this._setState(chunks(this._ordered()));
        if (this.menu.isOpen && this._busyMenu) {
            this._stale = true;       // the user is in a job's submenu: don't pull it out from under them
            return;
        }
        this._rebuildMenu();
    }

    _info(text, cls) {
        const it = new PopupMenu.PopupMenuItem(text, {reactive: false, can_focus: false});
        if (cls)
            it.label.add_style_class_name(cls);
        this.menu.addMenuItem(it);
        return it;
    }

    _rebuildMenu() {
        this.menu.removeAll();
        this._busyMenu = false;
        this._stale = false;
        const wall = Date.now() / 1000;
        if (this._summary)
            this._info(this._summary, 'gpuq-host');
        for (const v of this._ordered()) {
            if (!v.snap) {
                this._info(`${v.host} — unreachable`, 'gpuq-dim');
                continue;
            }
            const snap = v.snap;
            const now = snap.now + Math.max(0, wall - v.wall);   // the host's clock, moved on since the fetch
            let head = v.host;
            if (snap.paused)
                head += `  ·  ⏸ paused ${fmtDur(now - snap.paused.since)}, nobody may start`;
            this._info(head, 'gpuq-host');
            for (const c of cards(snap)) {
                this._info(`  ${cardTitle(c)}`, 'gpuq-card');
                if (!c.entries.length)
                    this._info('    queue empty', 'gpuq-dim');
                c.entries.forEach((e, i) => this._jobMenu(v.host, c, e, i, now));
                if (c.unqueued.length)
                    this._info(`    unqueued GPU use: ${c.unqueued.join('; ')}`, 'gpuq-amber');
                if (c.misplaced.length)
                    this._info(`    resident on the wrong card: ${c.misplaced.join(', ')}`, 'gpuq-amber');
                if (c.residents.length)
                    this._info(`    residents: ${c.residents.join(', ')}`, 'gpuq-dim');
                if (c.desktop.length)
                    this._info(`    desktop: ${c.desktop.join(', ')}`, 'gpuq-dim');
            }
        }
        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());
        const tui = new PopupMenu.PopupMenuItem('Open gpuq TUI');
        tui.connect('activate', () => {
            const argv = terminalArgv([gpuqPath(), 'tui']);
            if (!argv) {
                Main.notify('gpuq', 'No terminal found (tried ptyxis, gnome-terminal, kgx, konsole, xfce4-terminal).');
                return;
            }
            try {
                Gio.Subprocess.new(argv, Gio.SubprocessFlags.NONE);
            } catch (e) {
                Main.notify('gpuq', `Could not open a terminal: ${e.message}`);
            }
        });
        this.menu.addMenuItem(tui);
        const refresh = new PopupMenu.PopupMenuItem('Refresh now');
        refresh.connect('activate', () => {
            for (const v of this._views.values())
                v.at = -Infinity;
            this._poll(true);
        });
        this.menu.addMenuItem(refresh);
    }

    // One job's row: a submenu with move up/down/to next, and (confirmed) yield and remove.
    _jobMenu(host, c, e, i, now) {
        const sub = new PopupMenu.PopupSubMenuMenuItem(`    ${entryRow(e, i, now)}`);
        sub.menu.connect('open-state-changed', (m, open) => {
            if (open)
                this._busyMenu = true;
        });
        const n = c.entries.length, pos = i + 1;
        const headRunning = c.entries[0]?.state === 'running';
        const card = ['--gpu', c.name];
        const add = (text, args, confirm) => {
            const it = confirm ? new ConfirmItem(text, () => {
                this._busyMenu = true;
            }) : new PopupMenu.PopupMenuItem(text);
            it.connect('activate', () => this._act(host, args));
            sub.menu.addMenuItem(it);
        };
        if (e.state !== 'running') {
            const next = headRunning ? 2 : 1;            // right behind the running job: nothing is stopped
            if (pos - 1 > next)
                add(`Move to next (#${next})`, ['mv', e.id, String(next), ...card]);
            if (pos - 1 === 1 && headRunning)        // jumping above a running job asks it to yield
                add('Move up to #1 (asks the running job to yield)', ['mv', e.id, '1', ...card], true);
            else if (pos > 1)
                add(`Move up (#${pos - 1})`, ['mv', e.id, String(pos - 1), ...card]);
        }
        if (pos < n && e.state !== 'running')
            add(`Move down (#${pos + 1})`, ['mv', e.id, String(pos + 1), ...card]);
        if (e.state === 'running' && n > 1 && !e.yield_since)
            add('Ask to yield (it steps back to #2 and resumes later)', ['yield', e.id], true);
        add('Remove from the queue (does not stop the process)', ['rm', e.id, ...card], true);
        this.menu.addMenuItem(sub);
    }

    async _act(host, args) {
        const argv = host === this._local ? [gpuqPath(), ...args] : [gpuqPath(), '--host', host, ...args];
        const r = await actAsync(argv, host === this._local ? LOCAL_TIMEOUT_S : REMOTE_TIMEOUT_S);
        if (this._destroyed)
            return;
        if (!r.ok)
            Main.notify('gpuq', `${host}: gpuq ${args.join(' ')} failed${r.out ? `: ${r.out.split('\n').pop()}` : ''}`);
        const v = this._views.get(host);
        if (v)
            v.at = -Infinity;
        this._busyMenu = false;
        this._poll(true);
    }

    destroy() {
        this._destroyed = true;
        if (this._timer) {
            GLib.source_remove(this._timer);
            this._timer = 0;
        }
        super.destroy();
    }
});

export default class GpuqExtension extends Extension {
    enable() {
        this._indicator = new Indicator(this.path);
        Main.panel.addToStatusArea(this.uuid, this._indicator);
    }

    disable() {
        this._indicator?.destroy();
        this._indicator = null;
    }
}
