# make reproduce   rerun the out-of-sample backtest (about a minute), write results/ and re-render the README
# make check       verify README Key Results match results/ (fast, used in CI)
# make test        run the unit tests
.PHONY: setup reproduce render check test

setup:
	uv sync

reproduce:
	uv run python scripts/reproduce.py
	uv run python scripts/render_readme.py

render:
	uv run python scripts/render_readme.py

check:
	uv run python scripts/render_readme.py --check

test:
	MPLBACKEND=Agg uv run pytest -q
