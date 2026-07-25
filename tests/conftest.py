"""Pytest collection policy for the maintained test suite.

Desktop file synchronization left a few files named ``*conflicted copy*.py``
next to their maintained counterparts.  They are preserved for manual
reconciliation, but collecting both copies executes stale tests twice and can
mask which source is authoritative.
"""

collect_ignore_glob = ["*conflicted copy*.py"]
