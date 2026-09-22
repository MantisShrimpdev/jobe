"""Exact-law worlds: synthetic decisions whose answers are computed by code.

Each world module owns one JevBench family and shares `base.py` (the record,
its checks, the contamination guard, the CLI). Records carry the option each
named mistake produces, so a trained readout is graded on the law rather than
the surface. Built: `temporal_numeric.py` (eleven laws) and `long_policy.py`
(six laws). Not built: multi_hop.
"""
