import torch
from torch import Tensor
from typing import Iterable, Tuple, NamedTuple, Dict, TypeVar

def cat_tuples(*tuples: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
    return tuple(torch.cat(t) for t in zip(*tuples))

T = TypeVar('T', bound=NamedTuple)
def cat_namedtuples(*namedtuples: T) -> T:
    first = next(iter(namedtuples), None)
    if first is None:
        raise ValueError('No NamedTuple provided')
    return first.__class__(*cat_tuples(*namedtuples))  # type: ignore

def cat_dicts(*dicts: Dict[str, Tensor]) -> Dict[str, Tensor]:
    first = next(iter(dicts), None)
    if first is None:
        return {}
    return dict(zip(first.keys(), (torch.cat(a) for a in zip(*(d.values() for d in dicts)))))
