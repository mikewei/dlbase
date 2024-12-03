import torch
from torch import Tensor
from typing import Iterable, Tuple, NamedTuple, Dict

def cat_tuples(*tuples: Iterable[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
    return tuple(torch.cat(a) for a in zip(*tuples))

def cat_namedtuples(*namedtuples: Iterable[NamedTuple]) -> NamedTuple:
    return namedtuples[0].__class__(*cat_tuples(*namedtuples))

def cat_dicts(*dicts: Iterable[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    return dict(zip(dicts[0].keys(), (torch.cat(a) for a in zip(*(d.values() for d in dicts)))))
