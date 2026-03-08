from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import os
import types
import uuid
import weakref
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Iterator,
)
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
)
from typing import (
    Annotated,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from starlette.requests import HTTPConnection

from .exceptions import (
    AmbiguousServiceByTypeError,
    DuplicateServiceRegistrationError,
    InvalidServiceDefinitionError,
    ServiceContainerAccessError,
    ServiceContainerStateError,
    ServiceFactoryContractError,
    ServiceRegistrationError,
    UnregisteredServiceByKeyError,
    UnregisteredServiceByTypeError,
)
from .types import Ctor, Dtor, ServiceLifetime

logger = logging.getLogger(__name__)

_CURRENT_REQUEST_CTX: contextvars.ContextVar[HTTPConnection | None] = contextvars.ContextVar(
    "_svc_current_request",
    default=None,
)
_REQUEST_SCOPE_STATE_ATTR = "_fastapiex_di_request_scopes"
_SYNC_GENERATOR_NO_VALUE = object()


class _ContainerRequestScopeState:
    """Per-container request-local DI scope stacks."""

    __slots__ = ("cleanup_stack", "transaction_stack")

    def __init__(self) -> None:
        self.cleanup_stack: AsyncExitStack | None = None
        self.transaction_stack: AsyncExitStack | None = None

    def empty(self) -> bool:
        return self.cleanup_stack is None and self.transaction_stack is None


class _RequestScopeStateMap(
    weakref.WeakKeyDictionary["ServiceContainer", _ContainerRequestScopeState]
):
    """Opaque request.state mapping for per-container DI scope state."""


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
                raise ServiceContainerStateError(msg)
            # Enforce single-loop usage.
            container._ensure_event_loop()
            if container.destructing:
                msg = "Requesting service from a destructing container."
                logger.error(msg)
                raise ServiceContainerStateError(msg)

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
                    raise ServiceRegistrationError(msg)

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
        current_pid = os.getpid()
        if self._pid != current_pid:
            msg = (
                f"ServiceContainer accessed from different process "
                f"(created in PID {self._pid}, accessed from PID {current_pid}). "
                f"Each process must have its own ServiceContainer instance."
            )
            logger.error(msg)
            raise ServiceContainerAccessError(msg)

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
            raise ServiceContainerAccessError(msg) from exc

        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            msg = (
                "ServiceContainer used from multiple event loops; "
                "this is not supported. Create a separate container per loop."
            )
            logger.error(msg)
            raise ServiceContainerAccessError(msg)

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
                    origin, (AsyncIterator, Iterator)
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
        if origin is Awaitable:
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
        if origin is Coroutine:
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
            raise InvalidServiceDefinitionError(msg)

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
                raise ServiceRegistrationError(msg)
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
                    raise ServiceRegistrationError(msg)

        if service_type not in self._type_index:
            self._type_index[service_type] = set()
        self._type_index[service_type].add(internal_id)

    def _get_service_by_key(self, key: str) -> Service:
        """
        Resolve a service by its public key.

        Raises UnregisteredServiceByKeyError if no service is registered with the given key.
        """
        internal_id = self._key_index.get(key)
        if internal_id is None:
            msg = f"Requesting unregistered service: {key}"
            logger.debug(msg)
            raise UnregisteredServiceByKeyError(msg)
        return self._services[internal_id]

    def _get_service_by_type(self, service_type: object) -> Service:
        """
        Resolve a service by its registered type.

        Raises ServiceResolutionError if:
            - no service is registered for the given type; or
            - multiple services share the same type.
        """
        ids = self._type_index.get(service_type)
        if not ids:
            msg = f"No service registered for type: {service_type!r}"
            logger.debug(msg)
            raise UnregisteredServiceByTypeError(msg)
        if len(ids) > 1:
            msg = (
                f"Multiple services registered for type {service_type!r}; "
                f"use key-based injection instead."
            )
            logger.error(msg)
            raise AmbiguousServiceByTypeError(msg)
        internal_id = next(iter(ids))
        return self._services[internal_id]

    @staticmethod
    def _make_async_finalizer(
            dtor: Callable[[object], Awaitable[None]],
            instance: object,
    ) -> Callable[[], Awaitable[None]]:
        """
        Wrap a destructor into an async callable that can be awaited.
        """

        async def _finalizer() -> None:
            await dtor(instance)

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
            raise InvalidServiceDefinitionError(msg)
        if not inspect.iscoroutinefunction(dtor):
            msg = (
                f"Invalid destructor for service key={key!r}: "
                "destructor must be defined as async def."
            )
            logger.error(msg)
            raise InvalidServiceDefinitionError(msg)
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
            raise InvalidServiceDefinitionError(msg) from exc

    def _resolve_request_scope_state_map(
        self,
        request: HTTPConnection | None,
    ) -> _RequestScopeStateMap | None:
        if request is None:
            return None
        scope_state = getattr(request.state, _REQUEST_SCOPE_STATE_ATTR, None)
        if isinstance(scope_state, _RequestScopeStateMap):
            return scope_state
        return None

    def _get_or_create_request_scope_state_map(
        self,
        request: HTTPConnection,
    ) -> _RequestScopeStateMap:
        scope_state = self._resolve_request_scope_state_map(request)
        if scope_state is None:
            scope_state = _RequestScopeStateMap()
            setattr(request.state, _REQUEST_SCOPE_STATE_ATTR, scope_state)
        return scope_state

    def _get_or_create_container_request_scope_state(
        self,
        request: HTTPConnection,
    ) -> _ContainerRequestScopeState:
        state_map = self._get_or_create_request_scope_state_map(request)
        state = state_map.get(self)
        if state is None:
            state = _ContainerRequestScopeState()
            state_map[self] = state
        return state

    def _resolve_container_request_scope_state(
        self,
        request: HTTPConnection | None,
    ) -> _ContainerRequestScopeState | None:
        state_map = self._resolve_request_scope_state_map(request)
        if state_map is None:
            return None
        state = state_map.get(self)
        if isinstance(state, _ContainerRequestScopeState):
            return state
        return None

    def _resolve_cleanup_scope_stack(
        self,
        request: HTTPConnection | None,
    ) -> AsyncExitStack | None:
        state = self._resolve_container_request_scope_state(request)
        if state is None:
            return None
        stack = state.cleanup_stack
        if isinstance(stack, AsyncExitStack):
            return stack
        return None

    def _resolve_transaction_scope_stack(
        self,
        request: HTTPConnection | None,
    ) -> AsyncExitStack | None:
        state = self._resolve_container_request_scope_state(request)
        if state is None:
            return None
        stack = state.transaction_stack
        if isinstance(stack, AsyncExitStack):
            return stack
        return None

    def _bind_cleanup_scope_stack(
        self,
        request: HTTPConnection,
        stack: AsyncExitStack,
    ) -> None:
        state = self._get_or_create_container_request_scope_state(request)
        state.cleanup_stack = stack

    def _bind_transaction_scope_stack(
        self,
        request: HTTPConnection,
        stack: AsyncExitStack,
    ) -> None:
        state = self._get_or_create_container_request_scope_state(request)
        state.transaction_stack = stack

    def _clear_cleanup_scope_stack(
        self,
        request: HTTPConnection,
        *,
        expected: AsyncExitStack | None = None,
    ) -> None:
        state_map = self._resolve_request_scope_state_map(request)
        state = self._resolve_container_request_scope_state(request)
        if state_map is None or state is None:
            return
        current = state.cleanup_stack
        if expected is not None and current is not expected:
            logger.warning(
                "Cleanup scope stack mismatch on clear; keeping current stack binding."
            )
            return
        state.cleanup_stack = None
        if state.empty():
            state_map.pop(self, None)
        if not state_map and hasattr(request.state, _REQUEST_SCOPE_STATE_ATTR):
            delattr(request.state, _REQUEST_SCOPE_STATE_ATTR)

    def _clear_transaction_scope_stack(
        self,
        request: HTTPConnection,
        *,
        expected: AsyncExitStack | None = None,
    ) -> None:
        state_map = self._resolve_request_scope_state_map(request)
        state = self._resolve_container_request_scope_state(request)
        if state_map is None or state is None:
            return
        current = state.transaction_stack
        if expected is not None and current is not expected:
            logger.warning(
                "Transaction scope stack mismatch on clear; keeping current stack binding."
            )
            return
        state.transaction_stack = None
        if state.empty():
            state_map.pop(self, None)
        if not state_map and hasattr(request.state, _REQUEST_SCOPE_STATE_ATTR):
            delattr(request.state, _REQUEST_SCOPE_STATE_ATTR)

    def _require_cleanup_scope_stack(
        self,
        request: HTTPConnection | None,
        *,
        key_label: str,
    ) -> AsyncExitStack:
        stack = self._resolve_cleanup_scope_stack(request)
        if stack is not None:
            return stack
        msg = (
            f"Transient service '{key_label}' requires an active DI cleanup scope. "
            "Resolve it through Inject(...) before accessing cleanup-managed transients."
        )
        logger.error(msg)
        raise ServiceContainerAccessError(msg)

    def _require_transaction_scope_stack(
        self,
        request: HTTPConnection | None,
        *,
        key_label: str,
    ) -> AsyncExitStack:
        stack = self._resolve_transaction_scope_stack(request)
        if stack is not None:
            return stack
        msg = (
            f"Transient service '{key_label}' requires an active DI transaction scope. "
            "Resolve it through Inject(...) before accessing transactional transients."
        )
        logger.error(msg)
        raise ServiceContainerAccessError(msg)

    @staticmethod
    def _register_transient_dtor_on_stack(
        *,
        stack: AsyncExitStack,
        dtor: Callable[[object], Awaitable[None]],
        instance: object,
        key_label: str,
    ) -> None:
        async def _run_dtor() -> None:
            try:
                await dtor(instance)
            except Exception:
                logger.exception(
                    "Error running transient destructor for service '%s'.",
                    key_label,
                )

        stack.push_async_callback(_run_dtor)

    @staticmethod
    def _wrap_async_generator_result(
        agen: AsyncGenerator[object, object],
        *,
        key_label: str,
    ) -> AbstractAsyncContextManager[object]:
        @asynccontextmanager
        async def _managed() -> AsyncIterator[object]:
            try:
                instance = await agen.__anext__()
            except StopAsyncIteration:
                raise ServiceFactoryContractError(
                    f"Async contextmanager service '{key_label}' did not yield a value."
                ) from None

            try:
                yield instance
            except BaseException as exc:
                try:
                    await agen.athrow(exc)
                except StopAsyncIteration:
                    return
                except BaseException:
                    raise
                raise ServiceFactoryContractError(
                    f"Async contextmanager service '{key_label}' must yield exactly once."
                ) from None
            else:
                try:
                    await agen.__anext__()
                except StopAsyncIteration:
                    return
                raise ServiceFactoryContractError(
                    f"Async contextmanager service '{key_label}' must yield exactly once."
                )
            finally:
                await agen.aclose()

        return _managed()

    @staticmethod
    def _advance_sync_generator(
        gen: Generator[object, object, None],
    ) -> object:
        try:
            return next(gen)
        except StopIteration:
            return _SYNC_GENERATOR_NO_VALUE

    @staticmethod
    def _throw_sync_generator(
        gen: Generator[object, object, None],
        exc: BaseException,
    ) -> object:
        try:
            return gen.throw(exc)
        except StopIteration:
            return _SYNC_GENERATOR_NO_VALUE

    @staticmethod
    def _wrap_sync_generator_result(
        gen: Generator[object, object, None],
        *,
        key_label: str,
    ) -> AbstractAsyncContextManager[object]:
        @asynccontextmanager
        async def _managed() -> AsyncIterator[object]:
            try:
                instance = await asyncio.to_thread(
                    ServiceContainer._advance_sync_generator,
                    gen,
                )
            except Exception:
                await asyncio.to_thread(gen.close)
                raise

            if instance is _SYNC_GENERATOR_NO_VALUE:
                await asyncio.to_thread(gen.close)
                raise ServiceFactoryContractError(
                    f"Contextmanager service '{key_label}' did not yield a value."
                ) from None

            try:
                yield instance
            except BaseException as exc:
                try:
                    result = await asyncio.to_thread(
                        ServiceContainer._throw_sync_generator,
                        gen,
                        exc,
                    )
                except BaseException:
                    raise
                if result is _SYNC_GENERATOR_NO_VALUE:
                    return
                raise ServiceFactoryContractError(
                    f"Contextmanager service '{key_label}' must yield exactly once."
                ) from None
            else:
                result = await asyncio.to_thread(
                    ServiceContainer._advance_sync_generator,
                    gen,
                )
                if result is _SYNC_GENERATOR_NO_VALUE:
                    return
                raise ServiceFactoryContractError(
                    f"Contextmanager service '{key_label}' must yield exactly once."
                )
            finally:
                await asyncio.to_thread(gen.close)

        return _managed()

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
                    raise ServiceContainerStateError(msg)
                if service.instance is not None:
                    return service.instance

                await service.async_create_instance()
                logger.debug(f"Singleton service created: {key_label}")
                return service.instance

            # Transient resolution.
            ctor = service.ctor
            dtor = service.dtor

            # Execute factory: async directly, sync in a background thread to avoid blocking.
            if inspect.iscoroutinefunction(ctor):
                result = await ctor(*args, **kwargs)
            elif inspect.isasyncgenfunction(ctor):
                result = ctor(*args, **kwargs)
            else:
                # Run sync factories in a separate thread to avoid blocking the loop.
                result = await asyncio.to_thread(ctor, *args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result

            # Async contextmanager-style factory.
            if inspect.isasyncgen(result):
                transaction_stack = self._require_transaction_scope_stack(
                    request,
                    key_label=key_label,
                )
                agen = cast(AsyncGenerator[object, object], result)
                async_cm = self._wrap_async_generator_result(
                    agen,
                    key_label=key_label,
                )
                instance = await transaction_stack.enter_async_context(async_cm)
                if dtor is not None:
                    cleanup_stack = self._require_cleanup_scope_stack(
                        request,
                        key_label=key_label,
                    )
                    self._register_transient_dtor_on_stack(
                        stack=cleanup_stack,
                        dtor=dtor,
                        instance=instance,
                        key_label=key_label,
                    )
                return instance

            # Synchronous contextmanager-style factory.
            elif inspect.isgenerator(result):
                transaction_stack = self._require_transaction_scope_stack(
                    request,
                    key_label=key_label,
                )
                gen = cast(Generator[object, object, None], result)
                async_cm = self._wrap_sync_generator_result(
                    gen,
                    key_label=key_label,
                )
                instance = await transaction_stack.enter_async_context(async_cm)
                if dtor is not None:
                    cleanup_stack = self._require_cleanup_scope_stack(
                        request,
                        key_label=key_label,
                    )
                    self._register_transient_dtor_on_stack(
                        stack=cleanup_stack,
                        dtor=dtor,
                        instance=instance,
                        key_label=key_label,
                    )
                return instance

            # Plain object.
            else:
                instance = result
                if dtor:
                    cleanup_stack = self._require_cleanup_scope_stack(
                        request,
                        key_label=key_label,
                    )
                    self._register_transient_dtor_on_stack(
                        stack=cleanup_stack,
                        dtor=dtor,
                        instance=instance,
                        key_label=key_label,
                    )

            return instance
        finally:
            _CURRENT_REQUEST_CTX.reset(token)

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

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
            raise InvalidServiceDefinitionError(msg)

        async with self._lock:
            public_key = key
            internal_id = uuid.uuid4().hex

            if self.destructing:
                msg = "Cannot register services while container is destructing."
                logger.error(msg)
                raise ServiceContainerStateError(msg)

            if public_key is not None and public_key in self._key_index:
                msg = f"Duplicate service registration for key: {public_key}"
                logger.error(msg)
                raise DuplicateServiceRegistrationError(msg)

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
        into service-resolution calls, for request-scoped cleanup.
        """
        return f"_svc_request_{id(self)}"

    def current_request(self) -> HTTPConnection | None:
        """
        Return the currently resolving connection context, if any.
        """
        return _CURRENT_REQUEST_CTX.get()

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
