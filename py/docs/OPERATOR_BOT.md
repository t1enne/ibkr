# Operator — `ibkr bot`, unattended trading

The unattended path: one resident bot process per strategy config, running
`sync → cycle → sleep` on a cadence. The `bot` CLI only starts, stops and
reports; the trading logic is the same `src/live` cycle a scheduled `live run`
would execute, and the scope/lease/exit-code contracts are unchanged
(AGENTS.md § Live trading).

Run `uv run ibkr bot <command> --help` for the current defaults; this page
states the behaviour that does not drift.

## Commands

- `ibkr bot start <config.json ...>` — spawn one detached bot per config and
  return once every child claimed its scope (or exit non-zero). **Idempotent:**
  an already-live scope is reported as "already live" and never respawned.
  Options: `--dry-run` (print the plan; spawn nothing, write nothing),
  `--adapter ibkr|sim` (default: the config/precedence chooses), `-i/--interval`
  seconds (default: one bar of the config's `bars[0]`), `-a/--max-age` days
  (default: 2x the bar cadence, clamped `>= 1`), `--no-sync`, `--retries N`
  (transient cycles tolerated before the bot exits).
- `ibkr bot stop <config.json ...> [--all]` — request a clean stop and await
  exit. **Never signals a live bot**: it writes a stop flag the loop reads at
  its sleep point, so no cycle is interrupted. `--no-wait` writes the flag and
  returns; `--wait SECONDS` bounds the await — a bot still live at the bound is
  reported as such, not killed (escalate out of band). `stop` is not a flatten:
  the book stays open by design (D12/D13 stand).
- `ibkr bot status [--format text|json]` — one row per bot: scope, state
  (`live` / `exited` / `stale`), pid, started time, config path, last exit code.
  Exits non-zero when a still-live bot last reported a transient failure.

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

- `bot status` exits non-zero when a still-live bot last reported exit 1
  (mid-retry). Exited/stale bots appear in the table, never in the exit code —
  page on rows that do not come back after a scheduler pass.
- Alert on any non-zero `bot start` (a child failed to claim its scope) and on
  `bot status` exit 1 / exited or stale rows.
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
`runtime/` is gitignored). A cycle's exit code lands there every cycle. Logs
grow unbounded — rotation is out of band and required for a long-running
deployment.

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