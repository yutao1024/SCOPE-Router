"""Baseline router modules."""
from routers.baselines.oracle import Oracle
from routers.baselines.random_router import RandomRouter
from routers.baselines.strongest_global import StrongestGlobal
from routers.baselines.strongest_per_dataset import StrongestPerDataset
from routers.baselines.cheapest_global import CheapestGlobal

__all__ = [
    'Oracle',
    'RandomRouter',
    'StrongestGlobal',
    'StrongestPerDataset',
    'CheapestGlobal'
]

