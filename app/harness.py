"""Official entry template. Protocol frames only on stdout; logs on stderr."""
import sys

from contest_harness.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
