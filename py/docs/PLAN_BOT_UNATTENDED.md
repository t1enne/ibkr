# Plan — `ibkr bot`, unattended concurrent trading (MVP)

Subject: a non-blocking CLI that keeps N strategies trading unattended on N
broker targets — gateway checked, candles synced, cycles run on cadence — without
becoming a resident scheduler.

Design inputs: AGENTS.md (§ Live trading, § Alpha research), `src/live/lease.py`
(per-scope `flock`), `src/data/dl.py` (`download`), `src/data/ibkr/gateway.py`
(`ensure_ready`), todo.md D10/D12/D13/D14.

Status: plan only. No code written.

---

## 0. What already exists

One live cycle with machine exit codes (`src/live/cli.py`) and its pipeline
(`src/live/pipeline.py`), a per-scope OS `flock` lease (`src/live/lease.py`),
scope minting (`src/live/scope.py`), run-once candle download (`src/data/dl.py`),
and a gateway readiness seam (`src/data/ibkr/gateway.py::ensure_ready`).

Not shipped: no `bot` command (cron must hand-chain `data dl` → `live run`), no
concurrency surface, no process/pid management, no gateway singleton guard, and
D14 — the live path never downloads bars while `--max-age` defaults to 5 days.

---

## 1. The shape: registry + detached bots

`bot` never blocks and never trades. It splits:

- **Registry/supervisor** — `bot` starts, stops and reports _bots_. No trading
  logic, exits promptly.
- **Bot process** — one detached OS process per config (per scope) owning the
  loop: readiness → sync → cycle → sleep. Each cycle takes its own per-scope
  lease inside `live run`, so no new exclusion logic is introduced.

Repo posture is unchanged: there is no daemon, `bot` owns no restart policy.
systemd/cron re-runs an idempotent `bot start` after a reboot. Concurrency is
already blessed — the lease is per scope, so N configs on one host need nothing
new; the only constraint is one supervisor per scope, which is `bot`'s job.

```
bot start ──spawn──▶ bot A  (scope ibkr_a_…)
   │       ──spawn──▶ bot B  (scope ibkr_b_…)
   └─ probe gateway (fail closed if not ready)
bot status ──read registry──▶ liveness table
bot stop   ──SIGTERM──▶ bot …
```

---

## 2. Command surface

```
ibkr bot start <config.json ...>        # spawn detached bots; return immediately
ibkr bot stop <config.json ...> [--all] # SIGTERM, await exit
ibkr bot status [--format text|json]    # liveness table, per bot
```

`bot start` options: `-i/--interval SECONDS` (cycle cadence, default from
`bars[0]`), `-a/--max-age DAYS` (staleness gate, default from `bars[0]`, tightens
D14), `--no-sync`, `--adapter sim|ibkr`, `--dry-run` (spawn nothing; print the
plan and probe readiness), `-F/--format text|json`.

`bot start` is idempotent: a scope already live is reported as `live`, never
double-spawned.

---

## 3. Module layout

New package `src/bot/` — a bot supervisor is neither a live cycle nor gateway
lifecycle, so it belongs in neither `src/live` nor `src/gw`.

| File                  | Responsibility                                                |
| --------------------- | ------------------------------------------------------------- |
| `src/bot/cli.py`      | click `bot_group`; registered in `main.py` beside `live_group` |
| `src/bot/registry.py` | the runtime registry: liveness, list, claim                   |
| `src/bot/liveness.py` | the lock+pid identity primitive (libc seam)                   |
| `src/bot/spawn.py`    | detached `Popen` production `BotSpawner`                      |
| `src/bot/loop.py`     | the bot's own loop: cadence, retry, injected sleeper          |
| `src/bot/sync.py`     | derive symbols/bar/from → `src/data/dl.py::download`          |
| `src/bot/types.py`    | Protocols + frozen dataclasses                                |

`src/live`, `src/gw` and `src/db` are untouched. `src/db` stays a leaf; `bot`
reaches it only indirectly, via `live`.

---

## 4. Bot identity: liveness without pid reuse

A bare pid file is unsafe — pids are recycled and `bot stop` could SIGTERM an
unrelated process. Use the kernel primitive the lease already relies on.

- **Liveness = a held `flock` on `<db>.<scope>.bot.lock`**, held for the bot's
  entire lifetime. The kernel releases it on exit, crash or `SIGKILL`, so
  lock-held ⟺ alive and a stale _file_ is harmless. Same reasoning as
  `src/live/lease.py`; no TTL, no takeover rule.
- **The pid file is only a signal target** (`<db>.<scope>.bot.pid`). `bot stop`
  SIGTERMs that pid **only after confirming the lock is held and the pid's cmdline
  matches** — the lock proves _a_ bot owns the scope, the cmdline check rejects a
  recycled pid.
- **Logs:** `<runtime>/<scope>.log`.

`RuntimeDir` resolves file-relative (never `os.getcwd()`), like `src/db/path.py`,
and is overridable by env for tests.

```python
@dataclass(frozen=True)
class BotRecord:
    scope: str
    config_path: str
    pid: int
    started_at: pd.Timestamp
    state: Literal["live", "exited", "stale"]

class BotLiveness(Protocol):
    def probe(self, scope: str) -> BotRecord | None: ...
    def claim(self, scope: str) -> Iterator[None]: ...   # raises if held

class BotSpawner(Protocol):
    def spawn(self, argv: Sequence[str], log_path: Path) -> int: ...
```

---

## 5. Gateway assurance

MVP never manages the gateway: it **probes** `ensure_ready()`. Ready → proceed.
Not ready → fail closed (exit 1), spawn no bots, and tell the operator to run
`ibkr gw start`. `gw start` stays foreground and `src/gw` is untouched; detached
start and the singleton lock are deferred (see TODO).

Ordering is load-bearing: a bot is never spawned before the gateway is ready, and
a sync failure never reaches a cycle (§6).

---

## 6. The bot loop

Detached with `subprocess.Popen(argv, start_new_session=True, stdout=log,
stderr=STDOUT)` so `bot start` exits and the bot keeps trading. Numbered, in
order — each step fails closed:

1. `load_live_config(config_path)`.
2. Derive `max_age_days` and `interval` from `bars[0]` (D14: no loose 5-day default).
3. Claim the scope lock; held for life (§4).
4. `ensure_ready()`. Not ready → log, exit 1. **No sync, no cycle.**
5. Sync candles unless `--no-sync`:
   `download(cfg.symbols, from_date=now−warmup−buffer, bar=bars[0])`. Raise/timeout
   → exit 1. **No cycle** — a failed sync must never let the cycle trade a stale tail.
6. Run one cycle (`live run`'s orchestration, in-process). Its per-scope lease is
   taken inside; `bot` takes none itself.
7. Map the cycle exit code; append to the bot's log.
8. Sleep `interval` (injected `SleepFn`); repeat.

Retry: exit 1 (transient gateway/stale) retries up to a bounded count, then the
bot exits. Exit 2 and 3 stop the bot immediately. **One bot's exit never aborts
its siblings** — each is its own process.

---

## 7. Exit codes

| Bot exit | Cause                    | Action                     |
| -------- | ------------------------ | -------------------------- |
| 0        | clean cycle              | continue                   |
| 1        | config / stale / gateway | retry ≤N, then exit 1      |
| 2        | usage                    | exit 2                     |
| 3        | unsafe cycle             | exit 3 (needs an operator) |

`bot start` exits 0 once spawned (1 if the gateway is not ready, 2 on usage).
`bot status` exits the **most severe live bot code** so cron sees a wedged bot.
Per-bot codes land in that bot's log and its registry row.

---

## 8. Testing (`src/bot` — critical paths only)

1. Idempotent double `start` spawns a scope once.
2. A released lock (dead child) reads as `exited`/`stale`, not `live`.
3. `stop` refuses to SIGTERM when the lock is free or the cmdline mismatches.
4. Gateway not ready → no bot spawned, exit 1.
5. Sync raise → the cycle is never invoked (ordering, fail closed).
6. One bot's non-zero exit leaves sibling records untouched (isolation).
7. `--dry-run` spawns nothing and writes no registry row.

Not eligible (per AGENTS.md): strategy logic, candle gap math, report rendering,
real subprocesses, real sleeps, real network/IBKR.

**No migrations.** `bot` keeps zero durable state — cadence is wall clock + DB
newest bar, liveness is an OS lock, and the registry is a runtime dir of
`*.log` / `*.pid`.

---

## 9. Milestones

- **M1 — registry + spawn, no network.** `bot start --dry-run`, `bot status`, the
  liveness primitive, a faked `BotSpawner`. Pins idempotency, liveness, code
  isolation, dry-run writes nothing.
- **M2 — the bot loop.** cadence, retry policy, sync-then-gate ordering, injected
  sleeper.
- **M3 — gateway probe + docs.** bounded readiness check, cron/systemd unit, the
  D14 `--max-age` default.

---

## 10. Risks

- **`flock` is single-host / local-FS only** (inherited from `lease.py`) — one host
  per database, DB on local disk.
- **`bot stop` is not a flatten.** D12 (no kill switch) and D13 (no persisted
  equity) stand; stopping bots leaves the book open by design.
- **Supervisor is thin by intent.** No auto-restart: after a crash, `bot start`
  (idempotent) is the recovery, run by the scheduler.
- **Retry count and bounded readiness wait are tuning constants** — pick a
  default, expose a flag, keep numbers out of docstrings.

---

## TODO (one-liners)

- `bot logs <config> [-f]` — tail a bot log from the registry.
- Detached `ibkr gw start` + `gw.supervise.lock` singleton so `bot` can start the gateway.
- N-broker gateway instances (distinct ports/containers, per-instance lock) — after N-configs-on-one-gateway proves out.
- Log rotation (logrotate/systemd unit) rather than unbounded per-bot logs.
- `bot restart` as `stop` + `start` for one scheduler pass.
- Persisted NAV/drawdown table (D13) as its own rename-only migration.
- Distributed lock so N hosts can share one database.
- `--jitter` on cycle sleep to de-synchronize bots sharing a bar cadence.
