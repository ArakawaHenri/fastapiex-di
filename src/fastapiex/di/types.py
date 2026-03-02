from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from enum import IntEnum
from typing import Protocol, TypeAlias


class ServiceLifetime(IntEnum):
    """
    Lifetime of a registered service.

    SINGLETON:
        A single instance is created on first access and reused afterwards.
    TRANSIENT:
        A new instance is created for each request/resolution.
    """

    SINGLETON = 0
    TRANSIENT = 1


# Factory (ctor) may return:
# - a plain instance object
# - an Awaitable[object] (async def or sync returning awaitable)
# - a synchronous contextmanager-style factory via Iterator[object]
# - an asynchronous contextmanager-style factory via AsyncIterator[object]
Ctor: TypeAlias = Callable[
    ...,
    object | Awaitable[object] | Iterator[object] | AsyncIterator[object],
]

# Destructor (dtor): takes an instance object and must be async
Dtor: TypeAlias = Callable[[object], Awaitable[None]] | None


class CallableWithSignature(Protocol):
    __signature__: inspect.Signature
