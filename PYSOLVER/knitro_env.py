"""Helpers for the Knitro license and version.

Knitro looks for a license in the environment variable ``ARTELYS_LICENSE`` (the path of the license file or its
content) and in default locations, e.g. the ``Artelys`` folder under ``Program Files`` on Windows.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import knitro as kn


def prepare_license(license_file: str | Path | None = None) -> None:
    """Point ``ARTELYS_LICENSE`` to a license file for the current process only.

    The system settings are not modified.

    :param license_file: Path of the Knitro license file; if None, the environment is left unchanged.
    :raises FileNotFoundError: If ``license_file`` does not exist.
    """
    if license_file is not None:
        path = Path(license_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"license file not found: {path}")
        os.environ["ARTELYS_LICENSE"] = str(path)


def checkout_license() -> kn.License:
    """Check out a Knitro license token that can be shared by several problems.

    Sharing one token avoids a license round trip for every problem in a batch run.

    :return: The checked-out license token; call ``release()`` on it when done.
    :raises RuntimeError: If no valid license is found.
    """
    import knitro as kn

    lic = kn.License()  # does not fail without a license, so validate it by creating an empty problem
    try:
        kn.KN_free(kn.KN_new_lm(lic.lmc))
    except RuntimeError as err:
        lic.lmc = None  # avoid releasing an invalid token when the object is destroyed
        hint = (
            "Knitro could not find a valid license. Set the environment variable ARTELYS_LICENSE "
            "(note: underscore) to the license file path, or pass --license-file."
        )
        if "ARTELYS_LICENSE" not in os.environ and "ARTELYS LICENSE" in os.environ:
            hint += " Found a variable named 'ARTELYS LICENSE' (with a space): rename it to ARTELYS_LICENSE."
        raise RuntimeError(hint) from err
    return lic


def knitro_release() -> str:
    """Return the installed Knitro release.

    :return: The release string, e.g. ``"Knitro 16.0.0"``.
    """
    import knitro as kn

    return kn.KN_get_release()
