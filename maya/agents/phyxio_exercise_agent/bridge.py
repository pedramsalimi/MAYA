from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_phyxio_service():
    from maya.agents.phyxioAgent.phyxio_utils import PhyxioService

    return PhyxioService()

