from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .constants import SERVICE_DEFAULT_DESTROY_MARKER
from .registry import Require, Service, ServiceMap


class BaseService(ABC):
    """
    Base class for services.

    Singleton services should be named with the suffix "Service".
    Transient services should be named with the suffix "ServiceT".
    """

    @classmethod
    @abstractmethod
    def create(cls, *args: Any, **kwargs: Any) -> object:
        """Factory hook used by the service registry."""
        raise NotImplementedError

    @classmethod
    async def destroy(cls, instance: Any) -> None:  # noqa: B027
        """Optional cleanup hook. Override in services that need teardown."""
        _ = cls
        _ = instance

    setattr(destroy, SERVICE_DEFAULT_DESTROY_MARKER, True)


__all__ = ["BaseService", "Require", "Service", "ServiceMap"]
