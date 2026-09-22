"""TheLab records become valid jobe Decisions."""
from thelab.decisions.worlds.long_policy import generate as generate_policy
from thelab.decisions.worlds.temporal_numeric import generate as generate_temporal

from jobe.records import to_decision


def test_generated_records_become_valid_decisions():
    for w in generate_temporal(22, seed=1) + generate_policy(6, seed=1):
        d = to_decision(w)
        d.validate()
        assert [o.id for o in d.options] == [i for i, _ in w.options]
        d2 = to_decision(w.record())
        assert d2 == d
