# Plan — `ibkr bot`, unattended concurrent trading (MVP)

A non-blocking CLI keeping N strategies trading unattended on N broker targets:
candles synced, cycles run on cadence.

Inputs: AGENTS.md (§ Live trading), `src/live/lease.py`, `src/live/cli.py`
(`live_run`), `src/data/dl.py::download`, `src/exec/refs.py::scope_tag`, todo
D10/D12/D13/D14. Status: plan only.

**Posture change, deliberate.** "There is no daemon — you schedule it" still holds
for `live run`; `bot` is the exception: it spawns **N resident bot loops** (one
detached process per config, `sleep interval → cycle → repeat`). Only the
supervisor exits.

**The gateway is out of scope, entirely.** `bot` never starts, probes, supervises
or mentions it. Keeping a gateway reachable (`gw start`, systemd, tickle) is the
operator's job, done out of band. A cycle that hits a down gateway is just the
normal exit-1 retry path (§5).

Existing and relevant: the machine-exit-code cycle (`live_run`), its pipeline and
staleness gate, the per-scope `flock` lease, scope minting, run-once `data dl`.
Missing today: the `bot` command, process management, and D14 (the live path never
downloads bars while `--max-age` defaults loose).

---

## 1. The shape

- **Registry/supervisor** — `bot` starts, stops, reports _bots_. No trading logic,
  exits promptly.
- **Bot process** — one detached process per config (per scope): sync → cycle →
  sleep. Cycles keep their own per-scope lease inside `live run`, so no new
  exclusion logic. N configs on one host need nothing new; the only constraint is
  one supervisor per scope, which is `bot`'s job.
- **Restart policy: none.** After a crash the scheduler re-runs idempotent
  `bot start`.

```
bot start ──spawn──▶ bot A (scope ibkr_a_…)   bot status ──read registry──▶ table
          ──spawn──▶ bot B (scope ibkr_b_…)   bot stop  ──stop flag──▶ bot …  (§4)
```

---

## 2. Command surface

```
ibkr bot start <config.json ...>        # spawn detached bots; return immediately
ibkr bot stop <config.json ...> [--all] # request stop, await exit
ibkr bot status [--format text|json]    # liveness table, per bot
```

`start` options: `-i/--interval SECONDS` (cadence, default from `bars[0]`),
`-a/--max-age DAYS` (staleness gate; default **2× cadence in days, clamped
`>= 1`** — weekends cannot trip it and `<= 0` can never silently disable the gate,
D14), `--no-sync`, `--adapter sim|ibkr`, `-F/--format`, `--dry-run` (spawn
nothing; print the plan). `start` is idempotent (§4.1).

---

## 3. Module layout

New `src/bot/`: `cli.py` (`bot_group`, registered in `main.py`), `registry.py`
(liveness/list/claim/exit-code), `liveness.py` (lock+pid primitive, libc seam),
`spawn.py` (detached `Popen` `BotSpawner`), `loop.py` (cadence/retry/sleeper),
`sync.py` (derive symbols/bar/from → `download`), `types.py` (Protocols +
dataclasses). A bot supervisor is neither a live cycle nor gateway lifecycle.

Touched outside it, exactly two:

- **`src/live/run_once.py` (new):** the orchestration currently inlined in the
  `live_run` command (config load, adapter resolve, `SqliteLedger`, temp strategy
  projection, `asyncio.run`) moves behind a public `run_once(cfg, …) -> int`;
  `live_run` and `loop.py` both call it. Nothing else in `src/live` changes.
- **`src/main.py`:** register `bot_group`.

`src/db` stays a leaf; `bot` imports `src/db/path.py` directly (that module imports
nothing from `src.*`, so the leaf rule holds).

---

## 4. Bot identity

A bare pid file is unsafe — pids recycle, and a signal could hit an unrelated
process. Use the kernel primitive the lease already relies on.

- **Liveness = a held `flock` on `<db>.<scope_tag>.bot.lock`**, held for the bot's
  life. The kernel drops it on exit, crash or `SIGKILL`, so lock-held ⟺ alive and
  a stale _file_ is harmless — no TTL, no takeover. Filenames use
  `scope_tag(scope)`, the same derivation as the cycle lease, so they cannot drift.
- **The pid file is a record _and_ a signal target**
  (`<db>.<scope_tag>.bot.pid`): `pid`, `started_at`, `config_path`, last
  `exit_code`. A freed lock makes the row `exited`/`stale` regardless of the file.
- **`stop` never SIGTERMs a live bot.** It writes a stop flag the loop consumes at
  its sleep point, then awaits exit, so no cycle is killed between placement and
  `_persist` (step 7). Escalation is manual; a forced signal is still gated on
  lock-held + cmdline match.
- **Logs:** `<runtime>/<scope_tag>.log`. `RuntimeDir` resolves file-relative (never
  `os.getcwd()`) and is env-overridable for tests.

```python
@dataclass(frozen=True)
class BotRecord:
    scope: str
    config_path: str
    pid: int
    started_at: pd.Timestamp
    state: Literal["live", "exited", "stale"]
    exit_code: int | None

class BotLiveness(Protocol):
    def probe(self, scope: str) -> BotRecord | None: ...
    def claim(self, scope: str) -> Iterator[None]: ...   # raises if held

class BotSpawner(Protocol):
    def spawn(self, argv: Sequence[str], log_path: Path) -> int: ...
```

### 4.1 The spawn handshake

The lock is held by the **child**, the parent returns as soon as `Popen` returns,
and `start_new_session=True` does not carry the fd across — two concurrent starts
would both probe free and both spawn. So `start` takes a short-lived **spawning
claim** (`<db>.<scope_tag>.bot.spawn.lock`) around probe-and-spawn, and the child
signals on a pipe once it holds the scope lock. The parent reports `live` only
after that signal, and exits non-zero if any child failed to claim.

---

## 5. The bot loop

Detached via `Popen(argv, start_new_session=True, stdout=log, stderr=STDOUT)`.
Numbered, each step fails closed:

1. `load_live_config(config_path)`.
2. Derive `interval` and `max_age_days` from `bars[0]` (§2).
3. Claim the scope lock, held for life (§4).
4. Sync unless `--no-sync`: `download(cfg.symbols, from_date=date(...),
bar=bars[0])`, wrapped in bot-owned `asyncio.wait_for` (`download` has no
   timeout param); the signed warmup string is parsed bot-side into that `date`.
   Raise/timeout → exit 1. **No cycle** — a failed sync must never let the cycle
   trade a stale tail.
5. One cycle via `run_once(cfg, …)` (§3); its lease is taken inside, `bot` takes
   none. A down gateway is just a failing cycle here — exit 1, retried; `bot` does
   not look.
6. Map the exit code into the pid record **and** the log, EVERY cycle — not
   only at exit — so `status` sees a live bot's most recent outcome.
7. Sleep `interval` via an injected `SleepFn`, in short chunks; repeat. The stop
   flag is checked between chunks, never mid-cycle, so no live cycle is
   interrupted between placement and `_persist` (step 7) and `stop`'s bounded
   await can still observe the exit.

Exit 1 (transient stale/gateway) retries up to a bounded count then exits; exit 2
and 3 stop the bot immediately. **One bot's exit never aborts its siblings.**

---

## 6. Exit codes

| Bot exit | Cause                    | Action                     |
| -------- | ------------------------ | -------------------------- |
| 0        | clean cycle              | continue                   |
| 1        | config / stale / gateway | retry ≤N, then exit 1      |
| 2        | usage                    | exit 2                     |
| 3        | unsafe cycle             | exit 3 (needs an operator) |

`start` exits 0 once every child claimed its scope, else 1; 2 on usage. `status`
exits the most severe code a still-**live** bot reports in the same vocabulary.
In practice only `1` persists there: `2`/`3` exit the bot, so a bot is "live"
with one only for the instant before it drops its lock and reads `exited` — an
unsafe (3) or usage (2) bot is surfaced by the table, not by a lasting code.

---

## 7. Testing (`src/bot` — critical paths only)

1. Concurrent double `start` spawns a scope once.
2. A released lock (dead child) reads `exited`/`stale`, not `live` — the one test
   allowed to fork a real child; the primitive cannot be exercised in-process.
3. `stop` writes the flag and awaits; it never signals a live process.
4. Sync raise → the cycle is never invoked (ordering, fail closed).
5. One bot's non-zero exit leaves sibling records untouched.
6. `--dry-run` spawns nothing and writes no registry row.

Not eligible (AGENTS.md): strategy logic, candle gap math, report rendering, real
sleeps, real network/IBKR. **No migrations** — `bot` keeps zero durable state
(cadence is wall clock + DB newest bar, liveness is an OS lock, the registry is a
runtime dir of `*.log`/`*.pid`).

---

## 8. Milestones

- **M1 — registry + spawn, no network.** `start --dry-run`, `status`, the liveness
  primitive and spawn handshake, a faked `BotSpawner`. Pins idempotency, liveness,
  isolation, dry-run writes nothing.
- **M2 — the bot loop.** `run_once` extraction, cadence, retry policy,
  sync-then-cycle ordering, sleeper, stop flag.
- **M3 — docs.** The D14 `--max-age` default, the systemd unit for the bots, and
  the operator note that the gateway is theirs to keep alive.

---

## 9. Residuals

- **D10 inherited, not closed.** The gate still uses universe `MAX(timestamp)`, so
  one fresh symbol satisfies it; per-symbol freshness is out of scope.
- **Gateway availability is unmanaged by design** (§ top). `bot` cannot tell "my
  strategy broke" from "the gateway is down" — both are exit 1.
- **Log growth.** Per-bot logs are unbounded; rotation is out-of-band and required
  for long-running deployment.
- **`flock` is single-host / local-FS only** (inherited) — one host per database.
- **`stop` is not a flatten.** D12 (no kill switch) and D13 (no persisted equity)
  stand; stopping leaves the book open by design.
- **Retry count is a tuning constant** — default it, expose a flag, keep numbers
  out of docstrings.

---

## TODO (one-liners)

- `bot logs <config> [-f]` — tail a bot log from the registry.
- `bot restart` as `stop` + `start` for one scheduler pass.
- `--jitter` on cycle sleep to de-synchronize bots sharing a bar cadence.
- Per-symbol freshness gate (closes §9.1 / D10).
- Distinguish gateway-down from strategy failure in the exit-code map.
- Log rotation via logrotate/systemd unit (closes §9.3).
- N-broker support (N accounts, per-account scopes).
- Persisted NAV/drawdown table (D13) as its own rename-only migration.
- Distributed lock so N hosts can share one database.
