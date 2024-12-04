import pytest
import torch
from collections import namedtuple
from typing import NamedTuple
from ..tch import cat_tuples, cat_dicts, cat_namedtuples

def test_cat_tuples():
    assert cat_tuples() == ()
    assert cat_tuples((), ()) == ()
    t1 = (torch.tensor([1, 2]), torch.tensor([3, 4]))
    t2 = (torch.tensor([5, 6]), torch.tensor([7, 8]))
    t3 = (torch.tensor([1, 2, 5, 6]), torch.tensor([3, 4, 7, 8]))
    for a, b in zip(cat_tuples(t1, t2), t3):
        assert (a == b).all()

def test_cat_namedtuples():
    with pytest.raises(ValueError):
        cat_namedtuples()
    Empty = namedtuple('Empty', [])
    assert cat_namedtuples(Empty(), Empty()) == Empty()
    Test = namedtuple('Test', ['a', 'b'])
    t1 = Test(torch.tensor([1, 2]), torch.tensor([3, 4]))
    t2 = Test(torch.tensor([5, 6]), torch.tensor([7, 8]))
    t3 = Test(torch.tensor([1, 2, 5, 6]), torch.tensor([3, 4, 7, 8]))
    for a, b in zip(cat_namedtuples(t1, t2), t3):
        assert (a == b).all()
    class Test2(NamedTuple):
        a: torch.Tensor
        b: torch.Tensor
    t4 = Test2(torch.tensor([1, 2]), torch.tensor([3, 4]))
    t5 = Test2(torch.tensor([5, 6]), torch.tensor([7, 8]))
    t6 = Test2(torch.tensor([1, 2, 5, 6]), torch.tensor([3, 4, 7, 8]))
    for a, b in zip(cat_namedtuples(t4, t5), t6):
        assert (a == b).all()

def test_cat_dicts():
    assert cat_dicts() == {}
    assert cat_dicts({}, {}) == {}
    d1 = {'a': torch.tensor([1, 2]), 'b': torch.tensor([3, 4])}
    d2 = {'a': torch.tensor([5, 6]), 'b': torch.tensor([7, 8])}
    d3 = {'a': torch.tensor([1, 2, 5, 6]), 'b': torch.tensor([3, 4, 7, 8])}
    for (ak, av), (bk, bv) in zip(cat_dicts(d1, d2).items(), d3.items()):
        assert (ak == bk)
        assert (av == bv).all()
