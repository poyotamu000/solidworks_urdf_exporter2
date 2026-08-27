"""``strict_min_radius:`` -- where the fastener/bearing cut-off sits.

Inside an expanded sub-assembly a free rotation about a *small* concentric mate
is read as a screw in a clearance hole, not a hinge, because a 2 mm shaft is a
bolt far more often than it is a bearing.  The cut-off is 3 mm, which is right
for the machine-screw hardware that heuristic exists for -- and wrong for a
hobby serial-bus servo, whose output horn is a 2.4 mm boss.  A robot built from
those came out with every joint welded fixed until the cut-off could be moved.
"""
import pytest

from sw2robot.exporter.model import (
    _STRICT_MIN_RADIUS,
    classify_edge_auto,
    classify_edge_geo,
    strict_min_radius,
)

Z = [0.0, 0.0, 1.0]
O = [0.0, 0.0, 0.0]
CYL, PLANE = 4, 3


def _servo_horn(radius):
    """The mates an STS3215-style horn makes with its own case: one concentric
    boss plus the face it sits against -- a free spin about Z."""
    conc = {"type": "CONCENTRIC", "etypes": [CYL, CYL],
            "points": [O, O], "dirs": [Z, Z], "radii": [radius, radius]}
    face = {"type": "COINCIDENT", "etypes": [PLANE, PLANE],
            "points": [O, O], "dirs": [Z, Z], "radii": [None, None]}
    return [conc, face, conc, face]


def test_the_default_cut_off_welds_a_small_boss():
    jt, _ax, note = classify_edge_geo(_servo_horn(0.0024), strict=True)
    assert jt == "fixed"
    assert "fastener" in note


def test_lowering_it_lets_the_same_boss_turn():
    jt, ax, _note = classify_edge_geo(_servo_horn(0.0024), strict=True,
                                      min_radius=0.002)
    assert jt == "revolute"
    assert ax is not None


def test_a_boss_below_the_lowered_cut_off_still_welds():
    """Lowering the threshold moves it, it does not switch the rule off: an M1.6
    screw at 0.8 mm is still hardware."""
    jt, _ax, _note = classify_edge_geo(_servo_horn(0.0008), strict=True,
                                       min_radius=0.002)
    assert jt == "fixed"


def test_it_only_applies_in_strict_mode():
    """Top-level mates are not second-guessed by radius at all."""
    jt, _ax, _note = classify_edge_geo(_servo_horn(0.0024), strict=False)
    assert jt == "revolute"


def test_the_edge_record_carries_it_through_the_auto_classifier():
    """`_expand_one` stamps the configured value onto every edge it creates;
    this is the hand-off that has to survive."""
    rec = {"strict": True, "mates": _servo_horn(0.0024)}
    assert classify_edge_auto(rec)[0] == "fixed"
    rec["strict_min_radius"] = 0.002
    assert classify_edge_auto(rec)[0] == "revolute"


@pytest.mark.parametrize("value,expected", [
    (None, _STRICT_MIN_RADIUS), (0.002, 0.002), ("0.002", 0.002)])
def test_the_default_stands_in_for_an_unset_value(value, expected):
    assert strict_min_radius(value) == pytest.approx(expected)
