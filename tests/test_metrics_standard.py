import numpy as np

from evaluation.metrics_standard import (
    aji_kumar_greedy,
    aji_plus,
    binary_dice,
    pq_independent,
    pq_official,
    remap_label,
)


def test_remap_label_handles_gapped_ids():
    value = np.array([[0, 8, 8], [3, 0, 20]], dtype=np.int32)
    actual = remap_label(value)
    expected = np.array([[0, 2, 2], [1, 0, 3]], dtype=np.int32)
    np.testing.assert_array_equal(actual, expected)


def test_perfect_match_scores_one_with_noncontiguous_ids():
    true = np.array([[0, 2, 2], [0, 0, 7]], dtype=np.int32)
    pred = np.array([[0, 9, 9], [0, 0, 4]], dtype=np.int32)
    # The public HoVer-Net/PanNuke implementation divides SQ by TP + 1e-6.
    assert abs(pq_official(true, pred).pq - 1.0) < 1e-6
    assert pq_independent(true, pred).pq == 1.0
    assert aji_kumar_greedy(true, pred) == 1.0
    assert aji_plus(true, pred) == 1.0
    assert binary_dice(true, pred) == 1.0


def test_pq_threshold_is_strictly_greater_than_half():
    true = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.int32)
    pred = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.int32)
    # intersection=2, union=4, exactly 0.5: it must not be paired.
    assert pq_official(true, pred).tp == 0
    assert pq_independent(true, pred).tp == 0


def test_empty_conventions_are_explicit():
    empty = np.zeros((2, 2), dtype=np.int32)
    one = np.array([[1, 0], [0, 0]], dtype=np.int32)
    assert pq_official(empty, empty).pq == 1.0
    assert pq_official(empty, one).pq == 0.0
    assert pq_official(one, empty).pq == 0.0
    assert aji_kumar_greedy(empty, empty) == 1.0
    assert aji_plus(empty, empty) == 1.0
    assert binary_dice(empty, empty) == 1.0


def test_official_and_independent_pq_agree_on_split_merge_case():
    true = np.array(
        [[1, 1, 0, 2, 2], [1, 1, 0, 2, 2], [0, 0, 0, 0, 0]],
        dtype=np.int32,
    )
    pred = np.array(
        [[8, 8, 0, 4, 4], [8, 0, 7, 4, 4], [0, 0, 7, 0, 0]],
        dtype=np.int32,
    )
    official = pq_official(true, pred)
    independent = pq_independent(true, pred)
    assert abs(official.pq - independent.pq) < 1e-12
    assert official.tp == independent.tp
    assert official.fp == independent.fp
    assert official.fn == independent.fn
