from __future__ import annotations

import asyncio
import contextvars
import inspect
import os
import threading
import types
import uuid
import weakref
from collections.abc import AsyncIterator as AsyncIteratorABC
from collections.abc import Awaitable, Callable
from collections.abc import Awaitable as AwaitableABC
from collections.abc import Coroutine as CoroutineABC
from collections.abc import Iterator as IteratorABC
from enum import IntEnum
from typing import (
    Annotated,
    Protocol,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from fastapi.params import Depends
from loguru import logger
from starlette.requests import HTTPConnection
from starlette.types import ASGIApp, Receive, Scope, Send

REQUEST_FAILED_STATE_KEY = "_svc_request_failed"
_CURRENT_REQUEST_CTX: contextvars.ContextVar[HTTPConnection | None] = contextvars.ContextVar(
    "_svc_current_request",
    default=None,
)

# Factory (ctor) may return:
# - a plain instance object
# - an Awaitable[object] (async def or sync returning awaitable)
# - a synchronous contextmanager-style factory via Iterator[object]
# - an asynchronous contextmanager-style factory via AsyncIterator[object]
Ctor = Callable[
    ...,
    object | Awaitable[object] | IteratorABC[object] | AsyncIteratorABC[object],
]

# Destructor (dtor): takes an instance object, may be sync or async
Dtor = Callable[[object], None | Awaitable[None]] | None


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


class ServiceContainer:
    """
    Dependency container supporting key-based and type-based resolution,
    designed for async frameworks such as FastAPI.

    Concurrency model
    -----------------
    * The container is intended to be used **within a single asyncio event loop**.
    * All write operations (`register()`, `destruct_all_singletons()`) are
      serialized via an internal async lock (`self._lock`).
      The body of `register()` does **not await** while holding the lock, so
      updates are atomic with respect to other coroutines in the same loop.
    * Read operations (`aget_by_key()`, `aget_by_type()`) are lock-free for better
      throughput, but guarded by a lightweight event-loop consistency check.

    IMPORTANT:
    ----------
    * The container is **not thread-safe** and must not be shared across
      multiple event loops or OS threads.
    * While `destruct_all_singletons()` is running, `self.destructing` is set,
      and singleton resolution will fail fast with a clear error.
    """

    class SingletonService:
        """
        Internal representation of a singleton service.

        The instance is created once and cached.
        """

        __slots__ = (
            "_container_ref",
            "ctor",
            "dtor",
            "service_type",
            "public_key",
            "internal_id",
            "instance",
            "ctor_args",
            "ctor_kwargs",
            "_async_lock",
        )

        def __init__(
                self,
                container: ServiceContainer,
                ctor: Ctor,
                dtor: Dtor,
                service_type: object | None,
                public_key: str | None,
                internal_id: str,
                *args: object,
                **kwargs: object,
        ) -> None:
            self._container_ref: weakref.ReferenceType[ServiceContainer] = weakref.ref(
                container)
            self.ctor: Ctor = ctor
            self.dtor: Dtor = dtor
            self.service_type: object | None = service_type
            self.public_key: str | None = public_key
            self.internal_id: str = internal_id

            self.instance: object | None = None
            self.ctor_args: tuple[object, ...] = args
            self.ctor_kwargs: dict[str, object] = kwargs

            # Async lock for single-instance creation.
            self._async_lock = asyncio.Lock()

        def _ensure_container_active(self) -> None:
            """
            Ensure the owning container still exists and is not in destruction.
            Also enforces that we are running on the container's original loop.
            """
            container = self._container_ref()
            if container is None:
                msg = "Requesting service from a destroyed container."
                logger.error(msg)
                raise RuntimeError(msg)
            # Enforce single-loop usage.
            container._ensure_event_loop()
            if container.destructing:
                msg = "Requesting service from a destructing container."
                logger.error(msg)
                raise RuntimeError(msg)

        async def async_create_instance(self) -> None:
            """
            Create the singleton instance in an async context.

            - Coroutine factories are awaited directly.
            - Synchronous factories are executed in a background thread
              to avoid blocking the event loop.
            - Contextmanager-style factories (sync/async) are rejected
              (use TRANSIENT lifetime for those).
            """
            async with self._async_lock:
                if self.instance is not None:
                    return

                self._ensure_container_active()

                if inspect.iscoroutinefunction(self.ctor):
                    result = await self.ctor(*self.ctor_args, **self.ctor_kwargs)
                else:
                    # Run sync factories in a separate thread to avoid blocking the loop.
                    result = await asyncio.to_thread(
                        self.ctor,
                        *self.ctor_args,
                        **self.ctor_kwargs,
                    )
                    if inspect.isawaitable(result):
                        result = await result

                if inspect.isgenerator(result) or inspect.isasyncgen(result):
                    msg = (
                        "Contextmanager-style factory not allowed for "
                        "SINGLETON services. Use TRANSIENT lifetime instead."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

                self.instance = result

    class TransientService:
        """
        Internal representation of a transient service.

        A new instance is created on each resolution.
        """

        __slots__ = (
            "ctor",
            "dtor",
            "service_type",
            "public_key",
            "internal_id",
        )

        def __init__(
                self,
                ctor: Ctor,
                dtor: Dtor,
                service_type: object | None,
                public_key: str | None,
                internal_id: str,
        ) -> None:
            self.ctor: Ctor = ctor
            self.dtor: Dtor = dtor
            self.service_type: object | None = service_type
            self.public_key: str | None = public_key
            self.internal_id: str = internal_id

    # Type alias for internal service objects.
    Service = SingletonService | TransientService

    def __init__(self) -> None:
        import os

        # internal_id -> service
        self._services: dict[str, ServiceContainer.Service] = {}
        # public key -> internal_id
        self._key_index: dict[str, str] = {}
        # service_type -> set[internal_id]
        self._type_index: dict[object, set[str]] = {}

        # Guard against cross-loop / cross-thread access
        self._loop: asyncio.AbstractEventLoop | None = None

        # Guard against cross-process access
        self._pid: int = os.getpid()

        self.destructing: bool = False
        self._registrations_frozen: bool = False
        self._lock = asyncio.Lock()

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    def _ensure_same_process(self) -> None:
        """
        Ensure the container is used from the same process it was created in.

        This prevents issues in multi-process deployments (e.g., Gunicorn workers)
        where each worker should have its own container instance.
        """
        import os
        current_pid = os.getpid()
        if self._pid != current_pid:
            msg = (
                f"ServiceContainer accessed from different process "
                f"(created in PID {self._pid}, accessed from PID {current_pid}). "
                f"Each process must have its own ServiceContainer instance."
            )
            logger.error(msg)
            raise RuntimeError(msg)

    def _ensure_event_loop(self) -> asyncio.AbstractEventLoop:
        """
        Ensure the container is always used from the same event loop.

        This is a best-effort runtime check to catch accidental cross-loop
        access early rather than failing in subtle ways later.
        """
        # Check process isolation first
        self._ensure_same_process()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:  # pragma: no cover - defensive
            msg = "ServiceContainer methods must be used within an asyncio event loop."
            logger.error(msg)
            raise RuntimeError(msg) from exc

        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            msg = (
                "ServiceContainer used from multiple event loops; "
                "this is not supported. Create a separate container per loop."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        return loop

    @staticmethod
    def _infer_service_type(ctor: Ctor) -> object | None:
        """
        Infer the logical "service type" from the factory's return annotation.

        For non-contextmanager factories:
            - async def f() -> MyType: service type is MyType
            - def f() -> MyType: service type is MyType
            - def f() -> Awaitable[MyType] / Coroutine[..., MyType]:
              service type is MyType

        For sync contextmanager-style factories:
            - def f() -> Iterator[MyType]: service type is MyType

        For async contextmanager-style factories:
            - async def f() -> AsyncIterator[MyType]: service type is MyType

        If no meaningful annotation can be determined, returns None.

        The implementation is defensive and tries to be stable across Python
        versions and different typing styles (Annotated, Optional, | None, etc.).
        """
        try:
            sig = inspect.signature(ctor)
        except (TypeError, ValueError):
            return None

        ann = sig.return_annotation
        if ann is inspect.Signature.empty:
            return None

        # Resolve forward references / string annotations / Annotated
        try:
            try:
                closure_vars = inspect.getclosurevars(ctor)
                localns = closure_vars.nonlocals if closure_vars else None
            except Exception:
                localns = None

            hints = get_type_hints(ctor, include_extras=True, localns=localns)
            ann = hints.get("return", ann)
        except Exception:
            try:
                hints = get_type_hints(ctor, include_extras=True)
                ann = hints.get("return", ann)
            except Exception:
                # Type hints are best-effort only; never break runtime on failures.
                pass

        if isinstance(ann, str):
            resolved = ServiceContainer._resolve_annotation_str(ann, ctor)
            if resolved is None:
                logger.warning(
                    f"Unable to resolve forward reference '{ann}' to an actual type. "
                    "For anonymous registration, ensure all type annotations are resolvable."
                )
                return None
            ann = resolved

        # Unwrap Annotated[T, ...]
        if get_origin(ann) is Annotated:
            args = get_args(ann)
            ann = args[0] if args else None
            if ann is None:
                return None
            # Check if unwrapped type is a string forward reference
            if isinstance(ann, str):
                logger.warning(
                    f"Unable to resolve forward reference '{ann}' in Annotated type. "
                    "For anonymous registration, ensure all type annotations are resolvable."
                )
                return None

        origin = get_origin(ann)

        # Unwrap Optional[T] / Union[T, None] / T | None → T
        if origin is types.UnionType:
            u_args = [a for a in get_args(ann) if a is not type(None)]
            if len(u_args) == 1:
                ann = u_args[0]
                # Check if unwrapped type is a string forward reference
                if isinstance(ann, str):
                    logger.warning(
                        f"Unable to resolve forward reference '{ann}' in Optional/Union type. "
                        "For anonymous registration, ensure all type annotations are resolvable."
                    )
                    return None
                origin = get_origin(ann)

        # Async/sync contextmanager-style type hints.
        is_contextmanager_annotation = False
        if origin is not None:
            try:
                is_contextmanager_annotation = issubclass(
                    origin, (AsyncIteratorABC, IteratorABC)
                )
            except TypeError:
                is_contextmanager_annotation = False
        if is_contextmanager_annotation:
            args = get_args(ann)
            if args:
                inner_type = args[0]
                # Reject unresolved string forward references
                if isinstance(inner_type, str):
                    logger.warning(
                        f"Unable to resolve forward reference '{inner_type}' in contextmanager type. "
                        "For anonymous registration, ensure all type annotations are resolvable."
                    )
                    return None
                return cast(object, inner_type)
            return None

        # Awaitable[T]
        if origin is AwaitableABC:
            args = get_args(ann)
            if args:
                inner_type = args[0]
                # Reject unresolved string forward references
                if isinstance(inner_type, str):
                    logger.warning(
                        f"Unable to resolve forward reference '{inner_type}' in Awaitable type. "
                        "For anonymous registration, ensure all type annotations are resolvable."
                    )
                    return None
                logger.warning(
                    "Return annotation uses Awaitable[T]; prefer annotating async "
                    "factories as `-> T`. Using T as service type."
                )
                return cast(object, inner_type)

        # Coroutine[Any, Any, T]
        if origin is CoroutineABC:
            args = get_args(ann)
            if len(args) == 3:
                inner_type = args[2]
                # Reject unresolved string forward references
                if isinstance(inner_type, str):
                    logger.warning(
                        f"Unable to resolve forward reference '{inner_type}' in Coroutine type. "
                        "For anonymous registration, ensure all type annotations are resolvable."
                    )
                    return None
                logger.warning(
                    "Return annotation uses Coroutine[..., T]; prefer annotating async "
                    "factories as `-> T`. Using T as service type."
                )
                return cast(object, inner_type)

        return cast(object, ann)

    @staticmethod
    def _resolve_annotation_str(annotation: str, ctor: Ctor) -> object | None:
        try:
            closure_vars = inspect.getclosurevars(ctor)
            localns = closure_vars.nonlocals if closure_vars else None
        except Exception:
            localns = None

        try:
            return cast(
                object,
                eval(annotation, getattr(ctor, "__globals__", {}), localns or {}),
            )
        except Exception:
            return None

    def _register_type_index(
            self,
            service_type: object | None,
            internal_id: str,
            public_key: str | None,
    ) -> None:
        """
        Update internal type index and enforce registration rules for a given type.

        Anonymous registration:
            - Fails if any service (anonymous or named) of the same type already exists.
            - Fails if service_type is None (cannot infer type).

        Named registration:
            - Fails if an anonymous service of the same type already exists.
        """
        # Anonymous registration requires a valid service type
        if public_key is None and service_type is None:
            msg = (
                "Anonymous service registration failed: unable to infer service type. "
                "Please provide a type annotation on the factory function or use a named registration."
            )
            logger.error(msg)
            raise TypeError(msg)

        if service_type is None:
            return

        existing_ids = self._type_index.get(service_type, set())

        if public_key is None:
            # Anonymous registration: must be unique for this logical type.
            if existing_ids:
                msg = (
                    f"Anonymous registration for type {service_type!r} is not allowed: "
                    f"a service of this type already exists."
                )
                logger.error(msg)
                raise RuntimeError(msg)
        else:
            # Named registration: cannot coexist with an anonymous one of same type.
            for sid in existing_ids:
                svc = self._services.get(sid)
                if svc is not None and getattr(svc, "public_key", None) is None:
                    msg = (
                        f"Cannot register named service '{public_key}' for type {service_type!r}: "
                        f"a unique anonymous service of this type already exists."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

        if service_type not in self._type_index:
            self._type_index[service_type] = set()
        self._type_index[service_type].add(internal_id)

    def _get_service_by_key(self, key: str) -> Service:
        """
        Resolve a service by its public key.

        Raises RuntimeError if no service is registered with the given key.
        """
        internal_id = self._key_index.get(key)
        if internal_id is None:
            msg = f"Requesting unregistered service: {key}"
            logger.error(msg)
            raise RuntimeError(msg)
        return self._services[internal_id]

    def _get_service_by_type(self, service_type: object) -> Service:
        """
        Resolve a service by its registered type.

        Raises RuntimeError if:
            - no service is registered for the given type; or
            - multiple services share the same type.
        """
        ids = self._type_index.get(service_type)
        if not ids:
            msg = f"No service registered for type: {service_type!r}"
            logger.error(msg)
            raise RuntimeError(msg)
        if len(ids) > 1:
            msg = (
                f"Multiple services registered for type {service_type!r}; "
                f"use key-based injection instead."
            )
            logger.error(msg)
            raise RuntimeError(msg)
        internal_id = next(iter(ids))
        return self._services[internal_id]

    @staticmethod
    def _make_async_finalizer(
            dtor: Callable[[object], object | Awaitable[object]],
            instance: object,
    ) -> Callable[[], Awaitable[None]]:
        """
        Wrap a destructor into an async callable that can be awaited.

        The destructor itself may be synchronous or return an awaitable.
        """

        async def _finalizer() -> None:
            result = dtor(instance)
            if inspect.isawaitable(result):
                await result

        return _finalizer

    @staticmethod
    def _validate_destructor(dtor: object | None, key: str | None) -> None:
        if dtor is None:
            return
        if not callable(dtor):
            msg = (
                f"Invalid destructor for service key={key!r}: "
                f"expected a callable or None, got {type(dtor)!r}."
            )
            logger.error(msg)
            raise TypeError(msg)
        try:
            signature = inspect.signature(dtor)
        except (TypeError, ValueError):
            return
        try:
            signature.bind(object())
        except TypeError as exc:
            msg = (
                f"Invalid destructor signature for service key={key!r}: "
                "destructor must accept the service instance as a positional argument."
            )
            logger.error(msg)
            raise TypeError(msg) from exc

    async def _aget_impl(
            self,
            service: Service,
            request: HTTPConnection | None,
            key_label: str,
            *args: object,
            **kwargs: object,
    ) -> object:
        """
        Core async resolution logic shared by key-based and type-based resolution.
        """
        class _RequestFailedSignal(Exception):
            """Internal sentinel to drive contextmanager rollback on failed requests."""

        def _request_failed() -> bool:
            return (
                request is not None
                and bool(getattr(request.state, REQUEST_FAILED_STATE_KEY, False))
            )

        # Enforce single-loop usage for all resolution paths.
        self._ensure_event_loop()
        token = _CURRENT_REQUEST_CTX.set(request)
        try:
            if isinstance(service, ServiceContainer.SingletonService):
                # Singleton resolution.
                if args or kwargs:
                    logger.warning(
                        f"Arguments given for singleton service '{key_label}' are ignored."
                    )
                if self.destructing:
                    msg = f"Requesting service '{key_label}' while container is destructing."
                    logger.error(msg)
                    raise RuntimeError(msg)
                if service.instance is not None:
                    return service.instance

                await service.async_create_instance()
                logger.debug(f"Singleton service created: {key_label}")
                return service.instance

            # Transient resolution.
            ctor = service.ctor
            dtor = service.dtor
            finalizer: Callable[[], Awaitable[None]] | None = None

            # Execute factory: async directly, sync in a background thread to avoid blocking.
            if inspect.iscoroutinefunction(ctor):
                result = await ctor(*args, **kwargs)
            else:
                # Run sync factories in a separate thread to avoid blocking the loop.
                result = await asyncio.to_thread(ctor, *args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result

            # Async contextmanager-style factory.
            if inspect.isasyncgen(result):
                agen = result
                try:
                    instance = await agen.__anext__()
                except StopAsyncIteration:
                    raise RuntimeError(
                        f"Async contextmanager service '{key_label}' did not yield a value."
                    ) from None

                async def _close_gen() -> None:
                    """
                    Finalize a transient async contextmanager service using
                    async-contextmanager semantics (exactly one yield).

                    On success, advance once to completion so `else/finally` runs.
                    On failure, throw an internal signal so `except/finally` runs.
                    Any additional yielded value is treated as a contract violation.
                    """
                    try:
                        if _request_failed():
                            signal = _RequestFailedSignal()
                            try:
                                await agen.athrow(signal)
                            except StopAsyncIteration:
                                return
                            except _RequestFailedSignal:
                                return
                            raise RuntimeError(
                                f"Async contextmanager service '{key_label}' must yield exactly once."
                            )
                        try:
                            await agen.__anext__()
                        except StopAsyncIteration:
                            return
                        raise RuntimeError(
                            f"Async contextmanager service '{key_label}' must yield exactly once."
                        )
                    finally:
                        await agen.aclose()

                if dtor:
                    logger.warning(
                        f"Async contextmanager service '{key_label}' should use `async with` or "
                        f"`yield ... finally` for cleanup instead of a separate destructor."
                    )
                    dtor_finalizer = self._make_async_finalizer(dtor, instance)

                    async def _finalizer() -> None:
                        try:
                            await dtor_finalizer()
                        finally:
                            await _close_gen()

                    finalizer = _finalizer
                else:
                    finalizer = _close_gen

            # Synchronous contextmanager-style factory.
            elif inspect.isgenerator(result):
                gen = result
                try:
                    instance = next(gen)
                except StopIteration:
                    msg = f"Contextmanager service '{key_label}' did not yield a value."
                    logger.error(msg)
                    raise RuntimeError(msg)  # noqa: B904

                async def _close_gen() -> None:
                    """
                    Finalize a transient contextmanager service using
                    contextmanager semantics (exactly one yield).

                    On success, advance once to completion so `else/finally` runs.
                    On failure, throw an internal signal so `except/finally` runs.
                    Any additional yielded value is treated as a contract violation.
                    """

                    def _finalize_gen() -> None:
                        try:
                            if _request_failed():
                                signal = _RequestFailedSignal()
                                try:
                                    gen.throw(signal)
                                except StopIteration:
                                    return
                                except _RequestFailedSignal:
                                    return
                                raise RuntimeError(
                                    f"Contextmanager service '{key_label}' must yield exactly once."
                                )
                            try:
                                next(gen)
                            except StopIteration:
                                return
                            raise RuntimeError(
                                f"Contextmanager service '{key_label}' must yield exactly once."
                            )
                        except _RequestFailedSignal:
                            return
                        finally:
                            gen.close()

                    await asyncio.to_thread(_finalize_gen)

                if dtor:
                    logger.warning(
                        f"Contextmanager service '{key_label}' should use `with` or "
                        f"`yield ... finally` for cleanup instead of a separate destructor."
                    )
                    dtor_finalizer = self._make_async_finalizer(dtor, instance)

                    async def _finalizer() -> None:
                        try:
                            await dtor_finalizer()
                        finally:
                            await _close_gen()

                    finalizer = _finalizer
                else:
                    finalizer = _close_gen

            # Plain object.
            else:
                instance = result
                if dtor:
                    finalizer = self._make_async_finalizer(dtor, instance)

            if finalizer is not None:
                self._attach_finalizer_to_request(request, finalizer)

            return instance
        finally:
            _CURRENT_REQUEST_CTX.reset(token)

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    @property
    def registrations_frozen(self) -> bool:
        return self._registrations_frozen

    def freeze_registrations(self) -> None:
        """
        Freeze runtime registration updates on this container.

        Intended for production startup flow: register everything once,
        then reject accidental runtime mutation.
        """
        self._ensure_event_loop()
        self._registrations_frozen = True

    def unfreeze_registrations(self) -> None:
        """
        Allow runtime registration updates again.

        This is primarily intended for tests and controlled dev tooling.
        """
        self._ensure_event_loop()
        self._registrations_frozen = False

    async def register(
            self,
            key: str | None,
            lifetime: ServiceLifetime,
            ctor: Ctor,
            dtor: Dtor,
            *args: object,
            **kwargs: object,
    ) -> None:
        """
        Register a service in the container.

        This method is safe to call at runtime from within the same asyncio
        event loop. Registrations are serialized by an internal async lock.

        Parameters
        ----------
        key:
            Public key for the service. If None, the service is anonymous and
            can only be resolved by type. See class docstring for detailed rules.
        lifetime:
            Service lifetime (singleton or transient).
        ctor:
            Callable that creates the service instance. It may be synchronous,
            asynchronous, contextmanager-style (sync), or contextmanager-style (async) depending on
            the lifetime and usage.
        dtor:
            Optional destructor called when the service is torn down.
            For singletons, it is invoked by `destruct_all_singletons`.
            For transient services, it is invoked at the end of the request
            that created the instance.
        args, kwargs:
            Additional arguments passed to the factory for singleton services.
            For transient services, arguments must be supplied at resolution
            time; registration-time arguments are ignored with a warning.
        """
        # Enforce single-loop usage for all writes.
        self._ensure_event_loop()

        if not isinstance(lifetime, ServiceLifetime):
            msg = (
                f"Invalid lifetime for service key={key!r}: "
                f"expected ServiceLifetime, got {lifetime!r} ({type(lifetime)!r})."
            )
            logger.error(msg)
            raise TypeError(msg)

        async with self._lock:
            public_key = key
            internal_id = uuid.uuid4().hex

            if self._registrations_frozen:
                msg = "Cannot register services after container registrations are frozen."
                logger.error(msg)
                raise RuntimeError(msg)

            if self.destructing:
                msg = "Cannot register services while container is destructing."
                logger.error(msg)
                raise RuntimeError(msg)

            if public_key is not None and public_key in self._key_index:
                msg = f"Duplicate service registration for key: {public_key}"
                logger.error(msg)
                raise RuntimeError(msg)

            dtor_candidate: object | None = dtor
            self._validate_destructor(dtor_candidate, public_key)

            service_type = self._infer_service_type(ctor)

            if lifetime == ServiceLifetime.SINGLETON:
                service: ServiceContainer.Service = ServiceContainer.SingletonService(
                    self,
                    ctor,
                    dtor,
                    service_type,
                    public_key,
                    internal_id,
                    *args,
                    **kwargs,
                )
            else:
                if args or kwargs:
                    logger.warning(
                        "Arguments provided when registering a transient service are ignored."
                    )
                service = ServiceContainer.TransientService(
                    ctor,
                    dtor,
                    service_type,
                    public_key,
                    internal_id,
                )

            # Update indices first to avoid half-registered services on error.
            # This will raise an error if anonymous registration has no type
            self._register_type_index(service_type, internal_id, public_key)

            if public_key is not None:
                self._key_index[public_key] = internal_id

            # Finally insert into services table.
            self._services[internal_id] = service

            logger.debug(
                f"Service registered: key={public_key}, type={service_type}, id={internal_id}"
            )

    def request_kwarg_name(self) -> str:
        """
        Name of the keyword argument used when passing the connection object
        into service-resolution calls, for attaching transient finalizers.
        """
        return f"_svc_request_{id(self)}"

    def current_request(self) -> HTTPConnection | None:
        """
        Return the currently resolving connection context, if any.
        """
        return _CURRENT_REQUEST_CTX.get()

    def _request_ctx_key(self) -> str:
        """
        Name of the attribute on request.state used to store per-request data.

        A unique name based on the container id is used to avoid collisions.
        """
        return f"_svc_ctx_{id(self)}"

    def _get_or_create_request_ctx(
        self,
        request: HTTPConnection,
    ) -> dict[str, list[Callable[[], Awaitable[None]]]]:
        """
        Return the per-request context dictionary for this container.

        Structure:
            {
                "transient_finalizers": list[Callable[[], Awaitable[None]]]
            }
        """
        key = self._request_ctx_key()
        ctx: dict[str, list[Callable[[], Awaitable[None]]]] | None = getattr(
            request.state, key, None
        )
        if ctx is None:
            ctx = {"transient_finalizers": []}
            setattr(request.state, key, ctx)
        return ctx

    def _attach_finalizer_to_request(
            self,
            request: HTTPConnection | None,
            finalizer: Callable[[], Awaitable[None]],
    ) -> None:
        """
        Attach a transient finalizer to the current request.

        If request is None (e.g. resolution outside a request context),
        a warning is logged and the finalizer is not tracked.
        """
        if request is None:
            logger.warning(
                "No request context provided: transient finalizer cannot be attached "
                "(resource may leak)."
            )
            return
        ctx = self._get_or_create_request_ctx(request)
        ctx["transient_finalizers"].append(finalizer)

    async def aget_by_key(
        self,
        key: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        """
        Asynchronously resolve a service instance by key.

        This method is intended for internal or low-level use. In FastAPI
        endpoints, the `Inject()` helper should be preferred.
        """
        self._ensure_event_loop()

        _request_key = self.request_kwarg_name()
        request_raw = kwargs.pop(_request_key, None)
        request: HTTPConnection | None = (
            request_raw if isinstance(request_raw, HTTPConnection) else None
        )

        service = self._get_service_by_key(key)
        key_label = key
        return await self._aget_impl(service, request, key_label, *args, **kwargs)

    async def aget_by_type(
        self,
        service_type: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        """
        Asynchronously resolve a service instance by its registered type.

        Type-based resolution is only allowed when exactly one service of the
        given type is registered.
        """
        self._ensure_event_loop()

        _request_key = self.request_kwarg_name()
        request_raw = kwargs.pop(_request_key, None)
        request: HTTPConnection | None = (
            request_raw if isinstance(request_raw, HTTPConnection) else None
        )

        service = self._get_service_by_type(service_type)
        label = service.public_key or f"<type:{service_type!r}>"
        return await self._aget_impl(service, request, label, *args, **kwargs)

    async def destruct_all_singletons(self) -> None:
        """
        Destroy all singleton instances in reverse registration order.

        Each singleton's destructor (if any) is invoked and the instance
        reference is cleared. This method is idempotent and safe to call
        multiple times.

        While this method is running, the internal lock is held, so concurrent
        registration attempts will block until destruction finishes.
        """
        self._ensure_event_loop()

        async with self._lock:
            if self.destructing:
                return
            self.destructing = True
            logger.debug("Starting destruction of all singleton services...")

            try:
                for service in reversed(list(self._services.values())):
                    if (
                            isinstance(service, ServiceContainer.SingletonService)
                            and service.instance is not None
                    ):
                        label = service.public_key or service.internal_id
                        try:
                            if service.dtor:
                                finalizer = self._make_async_finalizer(
                                    service.dtor,
                                    service.instance,
                                )
                                await finalizer()
                        except Exception:
                            logger.exception(
                                f"Error destructing singleton service: {label}")
                        finally:
                            service.instance = None
                            logger.debug(
                                f"Singleton instance released: {label}")
            finally:
                self.destructing = False
                logger.debug("Finished destruction of all singleton services.")

    def require(
        self,
        key: str,
        *,
        allow_transient: bool = False,
    ) -> Callable[..., Awaitable[object]]:
        """
        Declare a dependency on another registered service by key.

        Returns a lazy async lambda that resolves the service when awaited.
        This is intended for wiring dependencies between services themselves,
        not for FastAPI endpoint resolution.
        """
        service = self._get_service_by_key(key)
        if not allow_transient and isinstance(service, ServiceContainer.TransientService):
            raise RuntimeError(
                f"Service '{key}' is transient. "
                "Singletons should not depend on transient services "
                "(set allow_transient=True to override)."
            )

        return lambda *a, **kw: self.aget_by_key(key, *a, **kw)


class ServiceContainerRegistry:
    """
    Thread-safe registry mapping the current (process, thread, event loop)
    execution context to a ServiceContainer.

    This lets one FastAPI app object host multiple independent containers when
    workers are implemented as threads (e.g., free-threaded runtimes).
    """

    def __init__(self) -> None:
        self._containers: dict[tuple[int, int, int], ServiceContainer] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _current_key() -> tuple[int, int, int]:
        loop = asyncio.get_running_loop()
        return (os.getpid(), threading.get_ident(), id(loop))

    def register_current(self, container: ServiceContainer) -> tuple[int, int, int]:
        key = self._current_key()
        with self._lock:
            existing = self._containers.get(key)
            if existing is not None and existing is not container:
                raise RuntimeError(
                    "A different ServiceContainer is already registered for the current event loop."
                )
            self._containers[key] = container
        return key

    def get_current(self) -> ServiceContainer | None:
        try:
            key = self._current_key()
        except RuntimeError:
            return None
        with self._lock:
            return self._containers.get(key)

    def unregister_current(
        self,
        *,
        expected: ServiceContainer | None = None,
    ) -> ServiceContainer | None:
        try:
            key = self._current_key()
        except RuntimeError:
            return None
        with self._lock:
            existing = self._containers.get(key)
            if existing is None:
                return None
            if expected is not None and existing is not expected:
                logger.warning(
                    "ServiceContainer registry mismatch on unregister; keeping current mapping."
                )
                return None
            return self._containers.pop(key)


_APP_STATE_REGISTRY_LOCK = threading.Lock()


class _AppStateWithRegistry(Protocol):
    sc_registry: ServiceContainerRegistry


class _CallableWithSignature(Protocol):
    __signature__: inspect.Signature


def get_or_create_service_container_registry(
    app_state: object,
) -> ServiceContainerRegistry:
    """
    Return the app-level ServiceContainerRegistry, creating it if necessary.
    """
    registry = getattr(app_state, "sc_registry", None)
    if isinstance(registry, ServiceContainerRegistry):
        return registry

    with _APP_STATE_REGISTRY_LOCK:
        registry = getattr(app_state, "sc_registry", None)
        if isinstance(registry, ServiceContainerRegistry):
            return registry
        registry = ServiceContainerRegistry()
        state = cast(_AppStateWithRegistry, app_state)
        state.sc_registry = registry
        return registry


def resolve_service_container(app_state: object) -> ServiceContainer | None:
    """
    Resolve the ServiceContainer for the current execution context.

    Resolution source:
    - loop-local registry on app.state (free-threaded safe)
    """
    if app_state is None:
        return None

    registry = getattr(app_state, "sc_registry", None)
    if isinstance(registry, ServiceContainerRegistry):
        return registry.get_current()
    return None


def Inject(
        target: str | type[object] | None = None,
        *args: object,
        **kwargs: object,
) -> Depends:
    """
    Create a FastAPI dependency marker for a registered service.

    In endpoints you can write:

        @router.get("/items")
        async def endpoint(db: DbConn = Inject(DbConn)):
            ...

    Resolution modes
    ----------------
    1. Key-based resolution:
        - Inject("my_key")

    2. Type-based resolution:
        - Inject(MyType)

       This requires that exactly one service of type MyType is registered.
    """
    # Decide lookup mode.
    lookup_value: str | type[object]
    if isinstance(target, str):
        key_specified = True
        lookup_value = target
    elif isinstance(target, type):
        key_specified = False
        lookup_value = target
    else:
        raise TypeError(
            "Inject() expects either a service key (str) or a service type."
        )

    # Build the dependency signature.
    params = [
        inspect.Parameter(
            "request",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=HTTPConnection,
        )
    ]
    pos_dep_map: dict[int, str] = {}
    pos_static: dict[int, object] = {}
    kw_dep_map: dict[str, str] = {}
    kw_static: dict[str, object] = {}

    for i, a in enumerate(args):
        if isinstance(a, Depends):
            name = f"_dep_arg_{i}_{uuid.uuid4().hex[:8]}"
            params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=a,
                )
            )
            pos_dep_map[i] = name
        else:
            pos_static[i] = a

    for k, v in kwargs.items():
        if isinstance(v, Depends):
            name = f"_dep_kw_{k}_{uuid.uuid4().hex[:8]}"
            params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=v,
                )
            )
            kw_dep_map[k] = name
        else:
            kw_static[k] = v

    sig = inspect.Signature(params)

    async def _dependency_callable(
        request: HTTPConnection,
        **resolved_deps: object,
    ) -> object:
        services = resolve_service_container(getattr(request.app, "state", None))
        if services is None:
            msg = (
                "Service container not initialized for the current event loop on FastAPI app state (sc_registry)."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        # Reconstruct positional arguments.
        final_args = [
            resolved_deps[pos_dep_map[i]
            ] if i in pos_dep_map else pos_static[i]
            for i in range(len(args))
        ]
        # Reconstruct keyword arguments.
        final_kwargs = {
            k: resolved_deps[kw_dep_map[k]
            ] if k in kw_dep_map else kw_static[k]
            for k in kwargs
        }

        # Attach request for tracking transient finalizers.
        final_kwargs[services.request_kwarg_name()] = request

        if isinstance(lookup_value, str):
            return await services.aget_by_key(lookup_value, *final_args, **final_kwargs)
        return await services.aget_by_type(lookup_value, *final_args, **final_kwargs)

    # Improve callable name for better error stacks & docs
    name_suffix = (
        str(lookup_value)
        if key_specified
        else getattr(lookup_value, "__name__", repr(lookup_value))
    )

    _dependency_callable.__name__ = (
        f"inject_{'key' if key_specified else 'type'}_{name_suffix}"
    )
    _dependency_callable.__qualname__ = _dependency_callable.__name__
    dependency_callable_with_signature = cast(
        _CallableWithSignature, _dependency_callable
    )
    dependency_callable_with_signature.__signature__ = sig

    return Depends(_dependency_callable)


class TransientServiceFinalizerMiddleware:
    """
    ASGI middleware that runs transient service finalizers at the end of
    each HTTP request or WebSocket connection.

    This middleware should be added to the ASGI app after all other middlewares
    that may resolve transient services.

    Usage:

        app.add_middleware(TransientServiceFinalizerMiddleware)
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def _run_finalizers(self, scope: Scope) -> None:
        app_state = getattr(scope.get("app"), "state", None)
        services = resolve_service_container(app_state)
        if services is None:
            return

        # scope["state"] is a plain dict, not the State wrapper object.
        # We must use dict operations directly.
        state = scope.get("state")
        if state is None:
            return
        ctx_key = services._request_ctx_key()
        ctx = state.get(ctx_key)
        if not ctx:
            return
        finalizers = ctx.get("transient_finalizers", [])
        for finalizer in reversed(finalizers):
            try:
                await finalizer()
            except Exception:
                logger.exception("Error running transient finalizer.")
        # Clean up the context from the state dict
        state.pop(ctx_key, None)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        state = scope.get("state")
        if state is None:
            state = {}
            scope["state"] = state

        try:
            await self.app(scope, receive, send)
        except Exception:
            state[REQUEST_FAILED_STATE_KEY] = True
            raise
        finally:
            # Run finalizers after response background tasks complete.
            await asyncio.shield(self._run_finalizers(scope))
