"""Quiet launcher probe, compatible with Windows PowerShell 5 and PowerShell 7."""

import sys


def main():
    if sys.version_info < (3, 10):
        return 1
    try:
        import numpy
    except (ImportError, OSError):
        return 1
    return 0 if numpy.__version__.split(".", 1)[0] == "2" else 1


if __name__ == "__main__":
    sys.exit(main())
