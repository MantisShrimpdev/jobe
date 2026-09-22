"""Windows shim for the Unix-only `fcntl` module, for LOCAL runs of the JevBench
harness only.

`jevbench/budget.py` takes an advisory `fcntl.flock` on its cost ledger so that
several runner processes can share one ledger safely. On Windows there is no
fcntl; this stub makes every lock a no-op.

That is safe here because a local conformance run is a single process with a
private ledger, so there is nothing to lock against. It changes NOTHING about
task loading, scoring, or results. Benchmark Heaven runs on Linux, where the
real module is used and this file is never on the path.

Activate by putting this directory FIRST on PYTHONPATH for the run, and only
then. Never install it globally.
"""

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8
F_GETFL = 3
F_SETFL = 4


def flock(fd, operation):  # noqa: ARG001 - signature parity with the real module
    return None


def lockf(fd, operation, length=0, start=0, whence=0):  # noqa: ARG001
    return None


def fcntl(fd, cmd, arg=0):  # noqa: ARG001
    return 0
