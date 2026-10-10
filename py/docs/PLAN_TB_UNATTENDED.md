# Plan — `ibkr tb`, unattended concurrent trading (rev 1)

Subject: a non-blocking CLI that keeps N strategies trading unattended on N
broker targets — gateway up, candles synced, cycles run on cadence — without
becoming a resident scheduler.

Design inputs: AGENTS.md (§ Live trading, § Alpha research), `src/live/lease.py`
(per-scope `flock`), `src/gw/supervise.py` (foreground supervisor),
`src/data/ibkr/sync.py` (`_compose_up` detached, `ensure_gateway_session`),
`src/data/ibkr/gateway.py` (`is_ready` / `ensure_ready`), `todo.md` D10/D12/D13/D14.

Status: plan only. No code written.

---

## 0. What already exists

| Piece                                             | Where                                                   | State                        |
| ------------------------------------------------- | ------------------------------------------------------- | ---------------------------- |
| One live cycle, cron-friendly, machine exit codes | `src/live/cli.py` (`live_run`)                          | shipped                      |
| Cycle pipeline, staleness gate (read-only)        | `src/live/pipeline.py` (`run_cycle`), `src/live/freshness.py` (`assert_data_fresh`) | shipped                      |
| Per-scope cycle lease (OS `flock`, non-blocking)  | `src/live/lease.py` (`file_lease`)                      | shipped                      |
| Scope minting `<adapter>_<config>_<hash>`         | `src/live/scope.py` (`scope_of`)                        | shipped                      |
| Run-once download of candles                      | `src/data/dl.py` (`download`, `data dl`)                | shipped                      |
| Gateway readiness seam                            | `src/data/ibkr/gateway.py` (`ensure_ready`)             | shipped                      |
| Gateway supervisor (tickle + 24h bounce)          | `src/gw/supervise.py` (`supervise_gateway`)             | shipped, **foreground only** |
| Detached compose up                               | `src/data/ibkr/sync.py` (`_compose_up`)                 | shipped                      |

What is NOT shipped, and is the subject of this plan:

- **No `tb` command.** Cron must today hand-chain `data dl` → `live run`.
- **No concurrency surface.** Nothing spawns or tracks N bots.
- **No process/pid management anywhere** in the repo (grep confirms).
- **No gateway singleton guard.** Two `gw start` processes would both tickle and
  both bounce; nothing prevents it.
- **`gw start` is foreground** — a `tb` that "ensures the gateway" cannot call it
  inline without blocking.
- **D14:** the live path never downloads bars; `--max-age` default 5 days is far
  looser than any bar cadence.

---

## 1. The shape: two layers

The earlier single-process `tb <cfg>` loop is rejected. `tb` must not block and
must run N bots concurrently, so it splits:

- **Layer 1 — registry/supervisor.** `tb` as a command that starts, stops and
  reports _bots_. It holds no trading logic and exits promptly.
- **Layer 2 — bot process.** One detached OS process per config (per scope). It
  owns the loop: readiness → sync → cycle → sleep. Each cycle takes its own
  per-scope lease inside `live run`, so no new exclusion logic is introduced.

This keeps the repo's "there is no daemon — you schedule it" posture: `tb` owns
no restart policy. systemd/cron re-runs an idempotent `tb start` after a reboot.

```
                 tb start ──spawn──▶ bot A  (scope ibkr_a_…)
                   │        ──spawn──▶ bot B  (scope ibkr_b_…)
                   │        ──spawn──▶ bot C  (scope sim_c_…)
                   └─ ensure gateway (singleton) ──▶ gw supervisor
                 tb status ──read registry──▶ liveness table
                 tb stop   ──SIGTERM──▶ bot …
```

### 1.1 Concurrency is already blessed

`src/live/lease.py` holds one lock **per scope**, not per database, explicitly so
"two adapters — `ibkr` and `sim` — or two configs must be able to run
CONCURRENTLY". N configs on one host therefore need nothing new; the constraint
is only that no _two supervisors of the same scope_ exist. That is Layer 1's job.

---

## 2. Command surface

```
ibkr tb start <config.json ...>          # spawn detached bots; return immediately
ibkr tb start --dir strats/pass          # every *.json in a directory
ibkr tb stop  [--scope SCOPE | --all]    # SIGTERM, await exit
ibkr tb status [--format text|json]      # liveness table, per bot
ibkr tb logs  --scope SCOPE [-f]         # tail one bot's log
```

`tb start` shared options:

| Flag                     | Meaning                                                              |
| ------------------------ | -------------------------------------------------------------------- |
| `-i/--interval SECONDS`  | cycle cadence; default derived from `bars[0]`                        |
| `-a/--max-age DAYS`      | staleness gate; default derived from `bars[0]` (tightens D14)        |
| `--no-sync`              | skip the candle sync step                                            |
| `-U/--universe PATH`     | override the download universe                                       |
| `--adapter sim\|ibkr`    | forwarded to `live run`                                              |
| `--mode paper\|live`     | **assertion only** — tb never starts a _session_, only the container |
| `--dry-run`              | spawn nothing; print the plan and probe readiness                    |
| `-F/--format text\|json` | report format                                                        |

`--allow-unsafe` is **not** offered on `tb`: it exists for report consumers and
would neutralise the only unsafe signal a scheduled run has.

`tb start` is idempotent: a scope already live is reported as `live`, never
double-spawned.

---

## 3. Module layout

New package `src/tb/` — a bot supervisor is neither a live cycle nor gateway
lifecycle, so it belongs in neither `src/live` nor `src/gw`.

| File                  | Responsibility                                                |
| --------------------- | ------------------------------------------------------------- |
| `src/tb/cli.py`       | click `tb_group`; registered in `main.py` beside `live_group` |
| `src/tb/registry.py`  | the on-disk bot registry: liveness, list, claim               |
| `src/tb/liveness.py`  | the lock+pid identity primitive (libc seam)                   |
| `src/tb/spawn.py`     | detached `Popen` production `BotSpawner`                      |
| `src/tb/gateway.py`   | `ensure_gateway_detached()` + singleton guard                 |
| `src/tb/loop.py`      | the bot's own loop: cadence, retry, injected sleeper          |
| `src/tb/sync.py`      | derive symbols/bar/from → `src/data/dl.py::download`          |
| `src/tb/readiness.py` | wraps `IbkrGateway.ensure_ready`                              |
| `src/tb/types.py`     | Protocols + frozen dataclasses                                |

`src/live` and `src/gw` are otherwise untouched except for the one singleton
lock in §5. `src/db` stays a leaf (tb reaches it only indirectly, via `live`).

---

## 4. Bot identity: liveness without pid reuse

A bare pid file is unsafe — pids are recycled, and `tb stop` could SIGTERM an
unrelated process. Use the same kernel primitive the lease already relies on.

- **Liveness = a held `flock` on `<db>.<scope>.tb.lock`.** The bot process opens
  it and holds it for its entire lifetime. The kernel releases it on exit,
  crash or `SIGKILL`, so lock-held ⟺ alive and a stale _file_ is harmless. Same
  reasoning as `src/live/lease.py`; no TTL, no takeover rule.
- **The pid file is only a signal target.** The bot writes
  `<db>.<scope>.tb.pid`, and `tb stop` SIGTERMs that pid **only after confirming
  the lock is held and the pid's cmdline matches** — the lock proves _a_ bot owns
  the scope, the cmdline check rejects a recycled pid.
- **Logs:** `<runtime>/<scope>.log`.

`RuntimeDir` resolves file-relative (never `os.getcwd()`), like
`src/db/path.py` and `src/config.py`, and is overridable by env for tests.

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

## 5. Gateway assurance (non-blocking)

`tb` must not block, and `gw start` is foreground — so `tb` never calls it
inline.

1. **Probe** `IbkrGateway.ensure_ready()`. Ready → proceed.
2. **Not ready** → spawn a detached `ibkr gw start` (own log), behind a **new
   singleton lock** `gw.supervise.lock` in `src/gw/`. Two supervisors would both
   tickle and both bounce the container; the lock makes a second `gw start`
   refuse rather than fight. This is the one small `src/gw/` addition.
3. **Still not ready** after a bounded wait → fail closed (exit 1), spawn no bots.

Ordering is load-bearing: a bot is never spawned before the gateway is ready, and
a sync failure never reaches a cycle (§7).

### 5.1 The N-broker question (open)

IBKR Client Portal is **one session = one account**. N _configs_ is trivial (N
scopes, one gateway). N _brokers/accounts_ needs N gateway stacks — distinct
ports and containers, each with its own singleton lock — which `gw` has no
instance concept for today. **Decision required before N-account support is
promised.** Recommend shipping N-configs-on-one-gateway first.

---

## 6. Spawn mechanics (the "non-blocking" contract)

```python
subprocess.Popen(argv, start_new_session=True, stdout=log, stderr=STDOUT)
```

`start_new_session=True` detaches from the controlling terminal (new session,
new process group), so `tb start` can exit and the bot keeps trading. `tb start`
returns once every bot is spawned, not once any cycle finishes.

---

## 7. The bot loop

Numbered, in order — each step fails closed:

1. `load_live_config(config_path)`.
2. Derive `max_age_days` and `interval` from `bars[0]` (D14: no loose 5-day default).
3. Claim the scope lock; held for life (§4).
4. `ensure_ready()`. Not ready → log, exit 1. **No sync, no cycle.**
5. Sync candles unless `--no-sync`: `download(cfg.symbols, from_date=now−warmup−buffer, bar=bars[0])`.
   Raise/timeout → exit 1. **No cycle** — a failed sync must never let the cycle
   trade a stale tail.
6. Run one cycle (`live run`'s orchestration, in-process). Its per-scope lease is
   taken inside; tb takes none itself.
7. Map the cycle exit code; append to the bot's log.
8. Sleep `interval` (injected `SleepFn`); repeat unless `--once`.

Retry: exit 1 (transient gateway/stale) retries up to a bounded count, then the
bot exits. Exit 2 and 3 stop the bot immediately. **One bot's exit never aborts
its siblings** — each is its own process.

---

## 8. Exit codes

| Bot exit | Cause                    | Action                     |
| -------- | ------------------------ | -------------------------- |
| 0        | clean cycle              | continue                   |
| 1        | config / stale / gateway | retry ≤N, then exit 1      |
| 2        | usage                    | exit 2                     |
| 3        | unsafe cycle             | exit 3 (needs an operator) |

`tb start` exits 0 once spawned (1 if the gateway never came ready, 2 on usage).
`tb status` exits the **most severe live bot code** so cron sees a wedged bot.
Per-bot codes land in that bot's log and its registry row.

---

## 9. Testing (`src/tb` — critical paths only)

Eligible:

1. Idempotent double `start` spawns a scope once.
2. A released lock (dead child) reads as `exited`/`stale`, not `live`.
3. `stop` refuses to SIGTERM when the lock is free or the cmdline mismatches.
4. Gateway not ready → no bot spawned, exit 1.
5. Sync raise → the cycle is never invoked (ordering, fail closed).
6. One bot's non-zero exit leaves sibling records untouched (isolation).
7. `status` aggregation returns the most severe live code.
8. `--dry-run` spawns nothing and writes no registry row.

Not eligible (per AGENTS.md): strategy logic, candle gap math,
`supervise_gateway` internals, report rendering, real subprocesses, real sleeps,
real network/IBKR.

---

## 10. Staging: new migrations?

**None.** `tb` keeps zero durable state — cadence is wall clock + DB newest bar,
liveness is an OS lock, and the registry is a runtime dir of `*.log` / `*.pid`.
Cycle bookkeeping that _would_ need a table (persisted NAV / drawdown, todo D13)
is deliberately deferred and must arrive as its own rename-only migration, not be
smuggled into `tb`.

---

## 11. Milestones

- **M1 — registry + spawn, no network.** `tb start --dry-run`, `tb status`, the
  liveness primitive, a faked `BotSpawner`. Pins idempotency, liveness, code
  isolation, dry-run writes nothing. No loop, no gateway.
- **M2 — the bot loop.** cadence, retry policy, sync-then-gate ordering,
  injected sleeper, `--once`.
- **M3 — gateway singleton + detached start.** `gw.supervise.lock`, bounded
  readiness wait, docs for the cron/systemd unit and the D14 `--max-age` default.
- **M4 (gated on §5.1)** — N-account gateway instances.

---

## 12. Risks and open questions

1. **N brokers vs N configs** (§5.1) — the largest unknown; blocks M4 only.
2. **`flock` is single-host / local-FS only** (inherited from `lease.py`) — one
   host per database, DB on local disk. N machines need a real distributed lock.
3. **`tb stop` is not a flatten.** D12 (no kill switch / flatten-all) and D13 (no
   persisted equity) stand; stopping bots leaves the book open by design.
4. **Log growth** — `tb` writes per-bot logs; rotation is an out-of-band
   logrotate/systemd concern, not implemented here.
5. **Supervisor is thin by intent.** No auto-restart, no supervision tree: after
   a crash, `tb start` (idempotent) is the recovery, run by the scheduler.
6. **Retry count and bounded readiness wait are tuning constants** — pick a
   default, expose a flag, and keep numbers out of docstrings.
