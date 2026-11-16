// gpuq top-bar indicator: pure functions over `gpuq show --json` snapshots (no gi imports, testable with gjs/node).

export function fmtDur(sec) {
    sec = Math.max(0, Math.round(sec));
    if (sec < 60)
        return `${sec}s`;
    const m = Math.floor(sec / 60);
    if (m < 60)
        return `${m}m`;
    const h = Math.floor(m / 60), mm = m % 60;
    return mm ? `${h}h${String(mm).padStart(2, '0')}` : `${h}h`;
}

export function trunc(s, n) {
    s = s ?? '';
    return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

// One summary per card on a host: {name, model, entries (line order), running, waiting, unqueued, residents, ...}.
export function cards(snap) {
    const out = [];
    const use = snap.card_use ?? {};
    const names = Object.keys(snap.lines ?? {}).sort((a, b) => (b === snap.default) - (a === snap.default));
    for (const card of names) {
        const ids = (snap.lines[card] ?? []).map(e => e.id);
        const byId = new Map((snap.entries ?? []).map(e => [e.id, e]));
        // lines hold the entries in queue order; fall back to the flat list if a line only has ids
        const entries = (snap.lines[card] ?? []).map(e => byId.get(e.id) ?? e);
        const running = entries.find(e => e.state === 'running') ?? null;
        const yielding = entries.some(e => e.yield_since);
        const waiting = entries.filter(e => e !== running).length;
        const u = use[card] ?? {};
        const jobs = u.jobs ?? [];
        const who = snap.who ?? {};
        out.push({
            name: card,
            model: snap.cards?.[card]?.model ?? card,
            mib: snap.cards?.[card]?.mib ?? null,
            meter: snap.meters?.[card] ?? null,
            entries, running, waiting, ids,
            unqueued: !running && !yielding && jobs.length > 0 ? jobs.map(j => who[j] ?? j) : [],
            residents: u.residents ?? [],
            desktop: u.desktop ?? [],
            misplaced: u.misplaced ?? [],
        });
    }
    return out;
}

// Panel chunks for every host: [{text, cls}]. A remote host with one card is shown by host name.
// view = {host, local, snap, err}.
export function chunks(views) {
    const out = [];
    for (const v of views) {
        if (!v.snap) {
            out.push({text: `${v.host}?`, cls: 'gpuq-unknown'});
            continue;
        }
        const cs = cards(v.snap);
        const paused = !!v.snap.paused;
        for (const c of cs) {
            const name = v.local ? c.name : cs.length === 1 ? v.host : `${v.host}:${c.name}`;
            // compact: "3090▶+2" running with 2 waiting, "3060·" idle; who is running is in the menu
            const behind = c.running ? c.waiting : c.entries.length;
            const text = `${name}${c.running ? '▶' : '·'}${behind > 0 ? `+${behind}` : ''}`;
            out.push({text, cls: c.unqueued.length || hot(c) ? 'gpuq-unqueued' : paused ? 'gpuq-paused' : ''});
        }
    }
    return out;
}

// One menu row for an entry at queue position i (0-based).
export function entryRow(e, i, now) {
    const bits = [];
    if (e.progress)
        bits.push(e.progress);
    if (e.state === 'running' && e.started) {
        const ran = now - e.started;
        bits.push(e.est_min ? `${fmtDur(ran)} of ~${fmtDur(e.est_min * 60)}` : `running ${fmtDur(ran)}`);
    } else {
        if (e.est_min)
            bits.push(`~${fmtDur(e.est_min * 60)}`);
        if (e.joined)
            bits.push(`waiting ${fmtDur(now - e.joined)}`);
    }
    if (e.max_min)
        bits.push(`max ${fmtDur(e.max_min * 60)}`);
    if (e.yield_since)
        bits.push(`yielding ${fmtDur(now - e.yield_since)}`);
    if (e.stalled)
        bits.push('stalled');
    const mark = e.state === 'running' ? '▶' : ' ';
    return `#${i + 1} ${mark} ${e.owner} — ${trunc(e.job, 48)}${bits.length ? `  ·  ${bits.join(' · ')}` : ''}`;
}

// Same thresholds as the TUI: core at 83 °C (NVIDIA starts slowing near there), GDDR6X memory at 100 °C.
export const HOT_C = 83, VRAM_HOT_C = 100;

export function hot(c) {
    return (c.meter?.temp_c ?? 0) >= HOT_C || (c.meter?.vram_temp_c ?? 0) >= VRAM_HOT_C;
}

export function cardTitle(c) {
    let t = c.model;
    if (c.mib)
        t += ` (${Math.round(c.mib / 1024)} GB)`;
    if (c.meter?.used_mib != null)
        t += `  ·  ${(c.meter.used_mib / 1024).toFixed(1)} GB used`;
    if (c.meter?.util != null)
        t += ` · ${c.meter.util}%`;
    if (c.meter?.temp_c != null)
        t += ` · ${c.meter.temp_c}°C`;
    if (c.meter?.vram_temp_c != null)
        t += ` · memory ${c.meter.vram_temp_c}°C`;
    if (hot(c))
        t += '  (hot)';
    return t;
}
