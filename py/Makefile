MAKEFLAGS += --no-print-directory --silent

SRC_DIRS := src/bt src/data src/exec src/indicators src/live src/research src/db src/shared
SRC      := $(SRC_DIRS) 
.PHONY: help lint format check test run exec

help:
	@echo "Usage:"
	@echo "  make run  <cli-args>  — Run the CLI"
	@echo "  make check            — Lint + format + typecheck + tests"
	@echo "  make test             — Run all tests (fast + db + slow)"
	@echo "  make lint             — Ruff check"
	@echo "  make format           — Ruff format"
	@echo "  make typecheck        — Ty check"
	@echo ""
	@echo "CLI commands via make run/exec:"
	@echo "  make run bt run <json>           — Run backtest"
	@echo "  make run bt run <json> --plot     — Run then open the dashboard"
	@echo "  make run bt run <json> -- --format jsonl  — JSON output"
	@echo ""
	@echo "Cli flags (-F, --help, etc.) must be separated from make args by '--':"
	@echo "  make run bt run strat.json -- --format jsonl"
	@echo "  make run bt optimize <json> '<grid>' --folds 4  — IS-tune → OOS validate"
	@echo "  make run data query <sym>        — Query candles"
	@echo "  make run data dl <sym>...        — Download candles"
	@echo "  make run bt --help               — Click-style help"
	@echo "  make run data --help             — Data subcommand help"

lint:
	uv run ruff check $(SRC)

format:
	uv run ruff format $(SRC)

typecheck:
	uv run ty check $(SRC)

test:
	uv run pytest $(SRC_DIRS) tests -v -m 'slow or not slow'

test-fast:
	uv run pytest $(SRC_DIRS) tests --no-header -q -m 'not slow and not db'

check: lint format typecheck test-fast

run:
	@if [ -n "$(ARGS)" ]; then \
		uv run ibkr $(ARGS); \
	else \
		uv run ibkr $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS)); \
	fi

exec: run

# Swallow extra words passed after make run so Make doesn't error.
# e.g. make run bt run strat.json → args consumed by 'run'; 'bt'/'run'/'strat.json' matched here.
%::
	@true
