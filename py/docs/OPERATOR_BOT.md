# Operator — `ibkr bot`, unattended trading

The unattended path: one resident bot process per strategy config, running
`sync → cycle → sleep` on a cadence. The `bot` CLI only starts, stops and
reports; the trading logic is the same `src/live` cycle a scheduled `live run`
would execute, and the scope/lease/exit-code contracts are unchanged
(AGENTS.md § Live trading).

Run `uv run ibkr bot <command> --help` for the current defaults; this page
states the behaviour that does not drift.

## Commands

Every command that prints results accepts `-F text|json`. Every command that
names a bot accepts a **TARGET**, which is any of: the config path, the bot's
NAME (the config basename without `.json`), the SCOPE, or an unambiguous SCOPE
prefix. A TARGET that matches two bots is a usage error (exit 2) listing the
candidates — it is never resolved by picking one.

- `ibkr bot start <config.json ...>` — spawn one detached bot per config and
  return once every child claimed its scope (or exit non-zero). **Idempotent:**
  an already-live scope is reported as "already live" and never respawned.
- **A config that no longer mints its row's scope is refused** (exit 1): the
  strategy-intent edit re-minted it, so the claim would protect one scope while
  the child trades another. `start`/`restart` name both scopes and tell you to
  stop the bot while flat, then start again — `--dry-run` reports the same
  refusal so a scheduler loop can be checked before it trips.
  Options: `--dry-run` (print the plan; spawn nothing, write nothing, create no
  log and take no lease), `--adapter ibkr|sim` (default: the
  config/precedence chooses), `-i/--interval` seconds (default: one bar of the
  config's `bars[0]`), `-a/--max-age` days (default: 2x the bar cadence,
  clamped `>= 1`), `--no-sync`, `--retries N` (transient cycles tolerated before
  the bot exits).
- `ibkr bot ls [TARGET ...] [-q]` (alias: `status`) — one row per bot the
  registry knows: name, scope, state (`live`/`exited`/`stale`), pid, last exit
  code, age and config path. **Exited and stale rows are listed by default** —
  the alerting contract pages on rows that never come back, so hiding them
  would hide exactly what to alert on; `--all` is therefore not a filter here.
  `-q` prints bare scopes for scripting (and the JSON form under `-F json`).
  Exits non-zero when a still-live bot last reported a transient failure.
- `ibkr bot inspect <TARGET ...>` — the resolved plan: name, scope, adapter,
  config, state, cadence (`interval`/`max_age`/`retries`/`sync`), the db, the
  lock/pid/stop-flag paths and the log path with its size. **Read-only.**
- `ibkr bot info` — where state lives (`DB`, `RUNTIME`) and which defaults and
  log settings apply, plus how many bots the registry holds per state.
  **Read-only.**
- `ibkr bot stop <TARGET ...> [--all]` — request a clean stop and await exit.
  **Never signals a live bot**: it writes a stop flag the loop reads at its
  sleep point, so no cycle is interrupted. `--no-wait` writes the flag and
  returns; `--wait SECONDS` bounds the await — a bot still live at the bound is
  reported as such, not killed (escalate out of band). `stop` is not a flatten:
  the book stays open by design (D12/D13 stand).
- `ibkr bot restart <TARGET ...> [--all] [--dry-run]` — stop then start each
  target. The stop is the same **request** as `bot stop` (`--wait SECONDS`
  bounds it); a bot still live at the bound is **not** force-killed and never
  replaced — that target reports `still live` and the command exits non-zero.
  A target that is already dead is simply started. `--all` restarts every bot
  the registry knows, by the identity recorded at spawn; a recorded config path
  gone from disk fails that row only. Start options are `start`'s: `--adapter`,
  `-i/--interval`, `-a/--max-age`, `--no-sync`, `--retries`, `--format`.
  `--dry-run` prints the plan and writes nothing — no spawn, no stop flag.
- `ibkr bot logs <TARGET ...> [--all]` — print each target's log, `docker logs`
  style. `-n/--tail N|all` (default: the last few hundred lines; `all` prints
  the file), `-f/--follow` streams appended lines until Ctrl-C, `--since
  15m|2h|1d` and `--level WARN` filter on the structured line, `-F text|json`
  gives lines or one object per line. **Read-only**: it writes nothing, takes no
  lock, spawns nothing. Several targets each get a `==> <TARGET> <==` header (a
  single target prints bare lines) and are read interleaved under `--follow`. A
  target with no log yet reports `no log for <TARGET> (<path>)` — as a JSON
  object under `-F json`; the command fails only when every target is missing.
  Because a TARGET may be a NAME or SCOPE, `logs` reaches a **running** bot even
  after its config was edited or deleted — the registry row owns the identity,
  so no scope is re-minted to read a log.

## Config-per-scope model

One config mints one scope (`<adapter>_<config_name>_<config_hash>`), and one
scope runs one resident bot. N configs on one host run concurrently; the
per-scope `flock` lease inside each cycle keeps the same scope from ever
overlapping. `flock` is single-host, local-FS — one host per database.

A strategy-intent edit (strategy type, symbols, params, bars, warm-up, sizing)
re-mints the scope and the old scope's book stops reconciling — the new scope
can re-enter the same position, doubling exposure. Edit strategy intent while
FLAT, exactly as for `live run`.

## Restart policy: none

A bot does not restart itself and nothing in the repo supervises it. A bounded
number of transient cycles end in exit 1 on purpose. If it must run
unattended, **the scheduler owns resurrection**: a cron line re-running
`ibkr bot start <config>` is the whole restart story — `start` is idempotent, so
a re-run is safe over a live bot. `scripts/ibkr-bot@.service` is the
start-on-boot half of that story. Note that with `RemainAfterExit=yes` the unit
stays active while the bot runs, so a systemd **timer cannot re-arm this same
template** — have the timer invoke `ibkr bot start` directly (or use a separate
oneshot unit).

## Exit codes and what to alert on

| code | meaning | operator action |
| ---- | ------- | ---------------- |
| 0 | clean cycle | none |
| 1 | config / stale data / gateway (transient) | the bot retries up to `--retries`, then exits 1 — the next scheduler `start` brings it back |
| 2 | usage | check the invocation |
| 3 | unsafe cycle | escalate now: the cycle may hold something the operator cannot see |

- `bot ls` (alias `bot status`) exits non-zero when a still-live bot last
  reported exit 1 (mid-retry). Exited/stale bots appear in the table, never in
  the exit code — page on rows that do not come back after a scheduler pass.
- Alert on any non-zero `bot start` (a child failed to claim its scope) and on
  `bot ls` exit 1 / exited or stale rows.
- Exit 3 and `DIVERGENCE`-class outcomes are the "operator escalates now"
  signal, from either `bot` or `live run`.

## The gateway is yours — entirely out of band

`bot` never starts, probes, supervises or mentions the gateway. Keeping a
gateway reachable and logged in (`ibkr gw`, your own systemd unit, tickle) is
the operator's job, outside `bot` and outside the unit in this repo. `bot
start` does not check it. A cycle that hits a down gateway is just the normal
exit-1 retry path — `bot` cannot tell "my strategy broke" from "the gateway is
down", so when exit 1 repeats, check the gateway first.

## Data freshness (D14)

Before every cycle the bot syncs the config's candles through the same
`data dl` primitive, wrapped in a bot-owned timeout; `--no-sync` opts out. The
staleness gate defaults to 2x the bar cadence in days, clamped `>= 1`, so it
trips on a genuine gap rather than on an ordinary cadence — but it keys off the
bar, not the trading calendar, so a `1d` bot over a long weekend can still see
its newest bar at the 2-day bound and take the self-healing exit-1 retry. The
clamp `>= 1` means `<= 0` can never silently disable the gate. An
explicit `-a/--max-age` always wins. `live run`'s own default is unchanged and a
manually scheduled `live run` still needs a `data dl` run first (todo D14 /
"No scheduling artifact").

## Logs

One log per bot: `<runtime>/<scope_tag>.log` (`IBKR_BOT_RUNTIME_DIR` overrides;
`runtime/` is gitignored). Every line the bot writes is one structured record:

```text
<ts> <LEVEL> <event> k=v…
2026-08-15T09:30:09Z WARN sync.fail err="timeout after 900s" scope=sim_x_71be04cd pid=24187
```

- `<ts>` is UTC ISO-8601 with seconds and a `Z`. It is the **line's own** time,
  so `tail -f` and `grep` read when the bot actually logged something — not when
  the file was last touched. It is the line's first field, so no flag re-prints
  it.
- `<LEVEL>` is `DEBUG`/`INFO`/`WARN`/`ERROR`; only `INFO` and `WARN` are
  emitted today (`ERROR`/`DEBUG` are reserved). `INFO` is lifecycle
  (`start.ready`, `cycle.exit`, `stop.observed`, `bot.exit`), `WARN` is
  recoverable (`sync.fail`, `retries.exhausted`). `DEBUG` is off unless
  `IBKR_BOT_LOG_LEVEL=DEBUG` and nothing emits it yet.
- `<event>` is a dotted, stable name. New names are added; existing ones are
  not renamed, so a saved `grep` keeps working across upgrades.
- `scope` and `pid` are ordinary fields on every line **the bot writes**, so
  output from several bots (`logs --all -f`) is self-identifying even when
  interleaved. A value with spaces, a quote or a backslash is double-quoted; `-`
  means absent (a literal `-` value is quoted so it cannot be read as absent).
  The child's stdout/stderr share this file, so a traceback lands there bare.

Each cycle logs `cycle.exit` with `code=`. That code is the same value written
to the pid row, in the same statement region — so `bot logs` and `bot ls EXIT`
can never disagree about a bot's last cycle. A log write that fails does not
change a cycle's exit code: the pid row stays the authority and the failure is
reported on stderr.

### Rotation

**The bot rotates its own log.** Before each append it checks the file's size
and, once it is over the bound, renames it to `.log.1` (shifting older
generations up and dropping the oldest). Nothing else has to run.

| env var | meaning |
| --- | --- |
| `IBKR_BOT_LOG_MAX_BYTES` | rotate at this size; `0` disables rotation |
| `IBKR_BOT_LOG_KEEP` | how many generations to keep beside the live file |
| `IBKR_BOT_LOG_LEVEL` | minimum level written to the file |

These are read by the **bot process at startup**, so change one and restart the
bot (`ibkr bot restart <TARGET>`). An out-of-band `logrotate` is still
survivable — ``logs --follow`` reopens a log that shrinks or is swapped — but
running both will double-rotate, so pick one; leave `logrotate` off if the
bot's own bound is enough.

### Reading

`ibkr bot logs` is the reader; nothing else writes there. `--follow` polls on a
short bounded sleep (no busy spin, Ctrl-C exits 0) and survives our own
rotation and an out-of-band one: a file that shrank, vanished or was replaced
is reopened from its start, so a rotation never strands the stream or replays
the rotated generation.

`--since 15m` and `--level WARN` filter on the parsed line. A line that does not
parse at all — a bare `cycle exit N` from an older build, a Python traceback, a
`src/live` diagnostic — is printed **verbatim** and is never filtered, so a
historical log still reads.

## systemd — `scripts/ibkr-bot@.service`

A template unit, one instance per config. See the comments in the unit for
the full contract; the short version:

1. Replace **every** `/path/to/ibkr/py` (`WorkingDirectory`, `ExecStart` and
   `ExecStop` — a missed `ExecStop` path means `systemctl stop` never runs
   `bot stop`, and with `KillMode=process` nothing else stops the bot) and set
   `BOT_CONFIG_DIR`. Install: `sudo cp scripts/ibkr-bot@.service
   /etc/systemd/system/` then `sudo systemctl daemon-reload`.
2. Start one bot per config, instance name = the config's basename:
   `sudo systemctl enable --now ibkr-bot@vwatr_hv_trendamp15.json`
   (bot for `strats/pass/vwatr_hv_trendamp15.json` when
   `BOT_CONFIG_DIR=strats/pass`).
3. Stop cleanly with `sudo systemctl stop ibkr-bot@<basename>` — the unit's
   `ExecStop` is `ibkr bot stop`, a flag request, never a signal.

The unit has `Restart=no`: the bot's exit-1-after-retries is a scheduled
condition, not a crash, and resurrection belongs to the scheduler that re-runs
idempotent `start`. It does not pass `--allow-unsafe` and does not touch the
gateway — keep it that way.