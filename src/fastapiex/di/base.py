from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .registry import Service, ServiceDict, require

_DEFAULT_DESTROY_MARKER = "__service_default_destroy_noop__"


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

    setattr(destroy, _DEFAULT_DESTROY_MARKER, True)


__all__ = ["BaseService", "Service", "ServiceDict", "require"]
