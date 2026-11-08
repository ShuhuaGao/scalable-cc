"""Check that Knitro and its license work by solving the Hock-Schittkowski problem 71 (HS71).

Usage: ``uv run python -m PYSOLVER.check_knitro [--license-file LICENSE_FILE]``
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .knitro_env import checkout_license, knitro_release, prepare_license

HS71_OPTIMUM = 17.0140173  # reference optimal objective value of HS71


def solve_hs71() -> tuple[int, float, list[float]]:
    """Solve HS71 with the Knitro expression API.

    :return: The Knitro status code, the objective value and the solution vector.
    :raises RuntimeError: If no valid Knitro license is found.
    """
    import knitro as kn

    lic = checkout_license()
    try:
        with kn.Problem(lic) as p:
            p.set_param("outlev", 0)
            x = [p.add_variable(lb=1.0, ub=5.0, x0=v) for v in (1.0, 5.0, 5.0, 1.0)]
            p.add_constraint(x[0] * x[1] * x[2] * x[3] >= 25.0)
            p.add_constraint(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 == 40.0)
            p.add_objective(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])
            rc = p.solve()
            return rc, kn.KN_get_obj_value(p.kc), [v.value for v in x]
    finally:
        lic.release()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the HS71 check from the command line.

    :param argv: Command-line arguments; ``sys.argv[1:]`` if None.
    :return: Process exit code, 0 if HS71 was solved to the reference optimum.
    """
    parser = argparse.ArgumentParser(description="Solve HS71 with Knitro to verify the installation and license.")
    parser.add_argument("--license-file", help="Knitro license file; sets ARTELYS_LICENSE for this process only")
    prepare_license(parser.parse_args(argv).license_file)
    try:
        rc, obj, x = solve_hs71()
    except RuntimeError as err:
        raise SystemExit(f"FAILED: {err}")
    ok = rc == 0 and abs(obj - HS71_OPTIMUM) < 1e-4
    print(f"{knitro_release()}: HS71 status {rc}, obj = {obj:.7f} (reference {HS71_OPTIMUM}), x = {[round(v, 5) for v in x]}")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
