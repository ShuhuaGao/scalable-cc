# Expert solver

The Python implementation uses Knitro 16 to solve the expert's nonlinear optimization problems. Install the project dependencies with `uv sync --locked`, install and license Knitro separately, and check the installation with `uv run python -m PYSOLVER.check_knitro`. Expert generation is configured by `configs/ideal10h_v1.json` and run with `scripts/run_experts_ideal10h.py`; see the experiment protocol for solver settings and reproduction instructions.
