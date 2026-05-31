from tcm_fuzzywiki.membership import Trapezoid, overlap_integral


def test_overlap_integral_identical_sets_near_one():
    shape = Trapezoid(0.2, 0.4, 0.8, 1.0)
    assert overlap_integral(shape, shape, points=300) > 0.70


def test_overlap_integral_disjoint_sets_near_zero():
    left = Trapezoid(0.0, 0.1, 0.2, 0.3)
    right = Trapezoid(0.7, 0.8, 0.9, 1.0)
    assert overlap_integral(left, right, points=300) == 0.0
