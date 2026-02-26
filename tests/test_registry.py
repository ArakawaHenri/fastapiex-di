from __future__ import annotations

from contextlib import contextmanager
from typing import cast

import pytest

from fastapiex.di import (
    BaseService,
    Require,
    Service,
    ServiceLifetime,
    ServiceMap,
)
from fastapiex.di.container import ServiceContainer
from fastapiex.di.registry import (
    AppServiceRegistry,
    RegisteredService,
    _topological_registration_order,
    build_service_plan,
    capture_service_registrations,
    register_services_from_registry,
)


@contextmanager
def _capture_here(registry: AppServiceRegistry):
    with capture_service_registrations(registry, include_packages=[__name__]):
        yield


def test_service_registration_outside_capture_is_allowed() -> None:
    @Service("outside_capture")
    class OutsideCaptureService(BaseService):
        @classmethod
        async def create(cls) -> OutsideCaptureService:
            return cls()

    assert OutsideCaptureService.__name__ == "OutsideCaptureService"


def test_service_map_expansion_and_require_template_resolution() -> None:
    registry = AppServiceRegistry()
    with _capture_here(registry):
        @Service("shared_dep")
        class SharedDepService(BaseService):
            @classmethod
            async def create(cls) -> SharedDepService:
                return cls()

        @ServiceMap("root_service", mapping={"alpha": {"name": "a"}, "beta": {"name": "b"}})
        class RootService(BaseService):
            @classmethod
            async def create(cls, name: str) -> RootService:
                _ = name
                return cls()

        @ServiceMap("{}_child_service", mapping={"alpha": {}, "beta": {}})
        class ChildService(BaseService):
            @classmethod
            async def create(
                cls,
                shared=Require("shared_dep"),
                scoped=Require("{}_root_service"),
            ) -> ChildService:
                _ = shared, scoped
                return cls()

    plan = build_service_plan(registry=registry)
    by_key = {spec.key: spec for spec in plan}

    assert "alpha_root_service" in by_key
    assert "beta_root_service" in by_key
    assert "alpha_child_service" in by_key
    assert "beta_child_service" in by_key

    alpha_deps = {
        dep.param_name: dep.required.target
        for dep in by_key["alpha_child_service"].dependencies
    }
    beta_deps = {
        dep.param_name: dep.required.target
        for dep in by_key["beta_child_service"].dependencies
    }

    assert alpha_deps["shared"] == "shared_dep"
    assert beta_deps["shared"] == "shared_dep"
    assert alpha_deps["scoped"] == "alpha_root_service"
    assert beta_deps["scoped"] == "beta_root_service"


@pytest.mark.asyncio
async def test_service_map_scalar_value_supports_positional_only_ctor_param() -> None:
    registry = AppServiceRegistry()
    with _capture_here(registry):
        @ServiceMap("{}_scalar", mapping={"alpha": "a", "beta": "b"})
        class ScalarService(BaseService):
            value: str

            def __init__(self, value: str) -> None:
                self.value = value

            @classmethod
            async def create(cls, value: str, /) -> ScalarService:
                return cls(value)

    container = ServiceContainer()
    await register_services_from_registry(container, registry=registry)

    alpha = await container.aget_by_key("alpha_scalar")
    beta = await container.aget_by_key("beta_scalar")
    assert isinstance(alpha, ScalarService)
    assert isinstance(beta, ScalarService)
    assert alpha.value == "a"
    assert beta.value == "b"


def test_service_map_rejects_positional_only_param_in_mapping_dict() -> None:
    registry = AppServiceRegistry()
    with _capture_here(registry):
        @ServiceMap("scalar", mapping={"alpha": {"value": "x"}})
        class BadScalarService(BaseService):
            @classmethod
            async def create(cls, value: str, /) -> BadScalarService:
                _ = value
                return cls()

    with pytest.raises(TypeError, match="positional-only"):
        build_service_plan(registry=registry)


def test_app_scoped_registry_isolated() -> None:
    registry = AppServiceRegistry()
    with _capture_here(registry):
        @Service("scoped_service")
        class ScopedService(BaseService):
            @classmethod
            async def create(cls) -> ScopedService:
                return cls()

    another = AppServiceRegistry()
    assert len(registry.definitions()) == 1
    assert another.definitions() == []


def test_capture_service_registrations_collects_into_app_registry() -> None:
    scoped_registry = AppServiceRegistry()

    with capture_service_registrations(
        scoped_registry,
        include_packages=[__name__],
    ):
        @Service("captured_service")
        class CapturedService(BaseService):
            @classmethod
            async def create(cls) -> CapturedService:
                return cls()

    scoped_keys = {
        definition.key_template for definition in scoped_registry.definitions()}
    assert "captured_service" in scoped_keys


def test_capture_service_registrations_respects_include_packages_filter() -> None:
    scoped_registry = AppServiceRegistry()

    with capture_service_registrations(
        scoped_registry,
        include_packages=["other.package"],
    ):
        @Service("filtered_out_service")
        class FilteredOutService(BaseService):
            @classmethod
            async def create(cls) -> FilteredOutService:
                return cls()

    assert scoped_registry.definitions() == []


@pytest.mark.asyncio
async def test_register_services_from_scoped_registry_only_uses_scoped_definitions() -> None:
    class ScopedPayload:
        pass

    scoped_registry = AppServiceRegistry()
    with _capture_here(scoped_registry):
        @Service("scoped_only_service")
        class ScopedOnlyService(BaseService):
            @classmethod
            async def create(cls) -> ScopedPayload:
                return ScopedPayload()

    container = ServiceContainer()
    await register_services_from_registry(
        container,
        registry=scoped_registry,
    )
    instance = await container.aget_by_key("scoped_only_service")
    assert isinstance(instance, ScopedPayload)


def test_topological_registration_order_detects_cycle() -> None:
    with pytest.raises(RuntimeError, match="circular"):
        _topological_registration_order(
            {
                "service_a": {"service_b"},
                "service_b": {"service_a"},
            }
        )


def test_service_lifetime_variants_and_eager_flag() -> None:
    registry = AppServiceRegistry()
    with _capture_here(registry):
        @Service("singleton_by_int", lifetime=0, eager=True)
        class SingletonByIntService(BaseService):
            @classmethod
            async def create(cls) -> SingletonByIntService:
                return cls()

        @Service("transient_by_int", lifetime=1)
        class TransientByIntService(BaseService):
            @classmethod
            async def create(cls) -> TransientByIntService:
                return cls()

        @Service("singleton_by_name", lifetime="Singleton")
        class SingletonByNameService(BaseService):
            @classmethod
            async def create(cls) -> SingletonByNameService:
                return cls()

        @Service("transient_by_name", lifetime="transient")
        class TransientByNameService(BaseService):
            @classmethod
            async def create(cls) -> TransientByNameService:
                return cls()

    plan = build_service_plan(registry=registry)
    by_key = {spec.key: spec for spec in plan}

    assert by_key["singleton_by_int"].lifetime == ServiceLifetime.SINGLETON
    assert by_key["singleton_by_int"].eager is True
    assert by_key["transient_by_int"].lifetime == ServiceLifetime.TRANSIENT
    assert by_key["singleton_by_name"].lifetime == ServiceLifetime.SINGLETON
    assert by_key["transient_by_name"].lifetime == ServiceLifetime.TRANSIENT


def test_eager_transient_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be eager with transient"):
        @Service("bad", lifetime="Transient", eager=True)
        class BadTransientService(BaseService):
            @classmethod
            async def create(cls) -> BadTransientService:
                return cls()


@pytest.mark.asyncio
async def test_service_decorator_supports_anonymous_registration_defaults() -> None:
    class AnonymousPayloadA:
        pass

    class AnonymousPayloadB:
        pass

    registry = AppServiceRegistry()
    with _capture_here(registry):
        @Service
        class AnonymousServiceA(BaseService):
            @classmethod
            async def create(cls) -> AnonymousPayloadA:
                _ = cls
                return AnonymousPayloadA()

        @Service()
        class AnonymousServiceB(BaseService):
            @classmethod
            async def create(cls) -> AnonymousPayloadB:
                _ = cls
                return AnonymousPayloadB()

    plan = build_service_plan(registry=registry)
    anonymous_specs = [spec for spec in plan if spec.key is None]
    assert len(anonymous_specs) == 2
    for spec in anonymous_specs:
        assert spec.lifetime == ServiceLifetime.SINGLETON

    container = ServiceContainer()
    registered = await register_services_from_registry(container, registry=registry)
    assert len(registered) == 2
    assert all(isinstance(item, RegisteredService) for item in registered)
    assert all(item.key is None for item in registered)
    assert all(item.origin.endswith(("AnonymousServiceA", "AnonymousServiceB"))
               for item in registered)

    a1 = await container.aget_by_type(AnonymousPayloadA)
    a2 = await container.aget_by_type(AnonymousPayloadA)
    b1 = await container.aget_by_type(AnonymousPayloadB)
    b2 = await container.aget_by_type(AnonymousPayloadB)

    assert a1 is a2
    assert b1 is b2


@pytest.mark.asyncio
async def test_named_service_can_depend_on_anonymous_service_by_type() -> None:
    class SharedPayload:
        pass

    registry = AppServiceRegistry()
    with _capture_here(registry):
        @Service
        class AnonymousSharedService(BaseService):
            @classmethod
            async def create(cls) -> SharedPayload:
                _ = cls
                return SharedPayload()

        @Service("consumer")
        class ConsumerService(BaseService):
            shared: SharedPayload

            def __init__(self, shared: SharedPayload) -> None:
                self.shared = shared

            @classmethod
            async def create(cls, shared=Require(SharedPayload)) -> ConsumerService:
                return cls(shared)

    container = ServiceContainer()
    await register_services_from_registry(container, registry=registry)

    consumer = await container.aget_by_key("consumer")
    shared = await container.aget_by_type(SharedPayload)
    assert isinstance(consumer, ConsumerService)
    assert consumer.shared is shared


@pytest.mark.asyncio
async def test_positional_only_required_dependency_is_injected_positionally() -> None:
    class SharedPayload:
        pass

    registry = AppServiceRegistry()
    with _capture_here(registry):
        @Service("dep_service")
        class DepService(BaseService):
            @classmethod
            async def create(cls) -> SharedPayload:
                return SharedPayload()

        @Service("consumer")
        class ConsumerService(BaseService):
            shared: SharedPayload

            def __init__(self, shared: SharedPayload) -> None:
                self.shared = shared

            @classmethod
            async def create(cls, shared=Require("dep_service"), /) -> ConsumerService:
                return cls(shared)

    container = ServiceContainer()
    await register_services_from_registry(container, registry=registry)

    consumer = await container.aget_by_key("consumer")
    shared = await container.aget_by_key("dep_service")
    assert isinstance(consumer, ConsumerService)
    assert consumer.shared is shared


def test_service_destroy_must_be_async_classmethod() -> None:
    with pytest.raises(TypeError, match="destroy must be an async @classmethod"):
        @Service("bad_destroy")
        class BadDestroyService(BaseService):
            @classmethod
            async def create(cls) -> BadDestroyService:
                return cls()

            @classmethod
            def destroy(cls, instance: object) -> None:
                _ = cls
                _ = instance


@pytest.mark.asyncio
async def test_anonymous_services_with_same_inferred_type_conflict_on_register() -> None:
    class SharedType:
        pass

    registry = AppServiceRegistry()
    with _capture_here(registry):
        @Service
        class AnonymousOne(BaseService):
            @classmethod
            async def create(cls) -> SharedType:
                _ = cls
                return SharedType()

        @Service()
        class AnonymousTwo(BaseService):
            @classmethod
            async def create(cls) -> SharedType:
                _ = cls
                return SharedType()

    container = ServiceContainer()
    with pytest.raises(RuntimeError, match="Anonymous registration for type"):
        await register_services_from_registry(container, registry=registry)


@pytest.mark.asyncio
async def test_inherited_destroy_hook_is_preserved() -> None:
    destroy_calls = 0

    class ParentService(BaseService):
        @classmethod
        async def create(cls) -> ParentService:
            return cls()

        @classmethod
        async def destroy(cls, instance: object) -> None:
            nonlocal destroy_calls
            _ = cls
            _ = instance
            destroy_calls += 1

    registry = AppServiceRegistry()
    with _capture_here(registry):
        @Service("child_service")
        class ChildService(ParentService):
            @classmethod
            async def create(cls) -> ChildService:
                return cls()

    plan = build_service_plan(registry=registry)
    by_key = {spec.key: spec for spec in plan}
    assert by_key["child_service"].dtor is not None

    container = ServiceContainer()
    await register_services_from_registry(container, registry=registry)
    await container.aget_by_key("child_service")
    await container.destruct_all_singletons()

    assert destroy_calls == 1


def test_require_string_key_cannot_target_anonymous_service() -> None:
    registry = AppServiceRegistry()
    with _capture_here(registry):
        @Service
        class AnonymousOnly(BaseService):
            @classmethod
            async def create(cls) -> AnonymousOnly:
                return cls()

        @Service("consumer")
        class ConsumerByKey(BaseService):
            @classmethod
            async def create(cls, dep=Require("anonymous_only")) -> ConsumerByKey:
                _ = dep
                return cls()

    with pytest.raises(RuntimeError, match="depends on 'anonymous_only'"):
        build_service_plan(registry=registry)


@pytest.mark.asyncio
async def test_transient_service_call_composition_prefers_defaulted_params() -> None:
    registry = AppServiceRegistry()
    with _capture_here(registry):
        @Service("flexible_transient", lifetime=ServiceLifetime.TRANSIENT)
        class FlexibleTransientService(BaseService):
            @classmethod
            async def create(
                cls,
                one: int,
                two: int = 2,
                *args: object,
                **kwargs: object,
            ) -> tuple[int, int, tuple[object, ...], dict[str, object]]:
                _ = cls
                return one, two, args, dict(kwargs)

    container = ServiceContainer()
    await register_services_from_registry(container, registry=registry)

    with pytest.raises(TypeError, match="missing required argument 'one'"):
        await container.aget_by_key("flexible_transient")

    by_kw = await container.aget_by_key("flexible_transient", one=1)
    by_pos = await container.aget_by_key("flexible_transient", 1)
    by_pos_with_extra = await container.aget_by_key("flexible_transient", 1, 3)
    by_kw_extra = await container.aget_by_key("flexible_transient", 1, three=3)
    by_kw_only_extra = await container.aget_by_key(
        "flexible_transient",
        one=1,
        three=3,
    )

    assert by_kw == (1, 2, (), {})
    assert by_pos == (1, 2, (), {})
    assert by_pos_with_extra == (1, 2, (3,), {})
    assert by_kw_extra == (1, 2, (), {"three": 3})
    assert by_kw_only_extra == (1, 2, (), {"three": 3})


@pytest.mark.asyncio
async def test_require_supports_args_and_kwargs_when_resolving_dependency() -> None:
    registry = AppServiceRegistry()
    with _capture_here(registry):
        @Service("arg_dep", lifetime=ServiceLifetime.TRANSIENT)
        class ArgDepService(BaseService):
            @classmethod
            async def create(
                cls,
                one: int,
                two: int = 2,
                *args: object,
                **kwargs: object,
            ) -> dict[str, object]:
                _ = cls
                return {
                    "one": one,
                    "two": two,
                    "args": args,
                    "kwargs": dict(kwargs),
                }

        @Service("consumer_with_arg_dep", lifetime=ServiceLifetime.TRANSIENT)
        class ConsumerWithArgDep(BaseService):
            @classmethod
            async def create(
                cls,
                dep=Require("arg_dep", 1, 3, three=3),
            ) -> object:
                _ = cls
                return dep

    container = ServiceContainer()
    await register_services_from_registry(container, registry=registry)

    payload = await container.aget_by_key("consumer_with_arg_dep")
    assert payload == {
        "one": 1,
        "two": 2,
        "args": (3,),
        "kwargs": {"three": 3},
    }


def test_servicemap_explicit_value_overrides_require_dependency_edge() -> None:
    registry = AppServiceRegistry()
    with _capture_here(registry):
        @ServiceMap("{}_consumer", mapping={"alpha": {"repo": "from_map"}})
        class ConsumerByMap(BaseService):
            @classmethod
            async def create(cls, repo=Require("missing_repo")) -> object:
                _ = cls
                return repo

    plan = build_service_plan(registry=registry)
    by_key = {spec.key: spec for spec in plan}
    consumer_spec = by_key["alpha_consumer"]
    assert consumer_spec.dependencies == ()


@pytest.mark.asyncio
async def test_singleton_missing_required_params_fails_during_registration() -> None:
    registry = AppServiceRegistry()
    with _capture_here(registry):
        @Service("bad_singleton")
        class BadSingleton(BaseService):
            @classmethod
            async def create(cls, one: int) -> BadSingleton:
                _ = one
                return cls()

    container = ServiceContainer()
    with pytest.raises(TypeError, match="Invalid singleton service 'bad_singleton'.*missing required argument 'one'"):
        await register_services_from_registry(container, registry=registry)


def test_servicemap_scalar_rejects_ambiguous_positional_binding_with_dependency_prefix() -> None:
    registry = AppServiceRegistry()
    with _capture_here(registry):
        @ServiceMap("{}_bad", mapping={"alpha": "v"})
        class BadScalarMapService(BaseService):
            @classmethod
            async def create(cls, dep=Require("dep"), value: str = "default", /) -> object:
                _ = cls, dep, value
                return object()

    with pytest.raises(TypeError, match="Scalar ServiceMap value cannot bind positional-only parameter"):
        build_service_plan(registry=registry)


@pytest.mark.asyncio
async def test_servicemap_scalar_prefers_keyword_target_and_keeps_require_dependency() -> None:
    class DepPayload:
        pass

    registry = AppServiceRegistry()
    with _capture_here(registry):
        @Service("dep")
        class DepService(BaseService):
            @classmethod
            async def create(cls) -> DepPayload:
                _ = cls
                return DepPayload()

        @ServiceMap("{}_consumer", mapping={"alpha": "from_map"})
        class ConsumerService(BaseService):
            @classmethod
            async def create(
                cls,
                dep=Require("dep"),
                value: str = "default",
            ) -> dict[str, object]:
                _ = cls
                return {"dep": dep, "value": value}

    plan = build_service_plan(registry=registry)
    by_key = {spec.key: spec for spec in plan}
    consumer_spec = by_key["alpha_consumer"]
    dep_names = {dep.param_name for dep in consumer_spec.dependencies}
    assert dep_names == {"dep"}

    container = ServiceContainer()
    await register_services_from_registry(container, registry=registry)
    payload = await container.aget_by_key("alpha_consumer")
    payload = cast(dict[str, object], payload)
    assert payload["value"] == "from_map"
    assert isinstance(payload["dep"], DepPayload)


@pytest.mark.asyncio
async def test_require_template_renders_target_and_args_kwargs_for_servicemap() -> None:
    registry = AppServiceRegistry()
    with _capture_here(registry):
        @ServiceMap("{}_dep", mapping={"alpha": {"name": "dep-a"}, "beta": {"name": "dep-b"}})
        class NamedDep(BaseService):
            @classmethod
            async def create(cls, name: str) -> dict[str, str]:
                _ = cls
                return {"name": name}

        @ServiceMap("{}_formatter", mapping={"alpha": {}, "beta": {}}, lifetime=ServiceLifetime.TRANSIENT)
        class FormatterService(BaseService):
            @classmethod
            async def create(cls, prefix: str, suffix: str = "") -> str:
                _ = cls
                return f"{prefix}{suffix}"

        @ServiceMap("{}_consumer", mapping={"alpha": {}, "beta": {}}, lifetime=ServiceLifetime.TRANSIENT)
        class ConsumerService(BaseService):
            @classmethod
            async def create(
                cls,
                dep=Require("{}_dep"),
                label=Require("{}_formatter", "{}", suffix="-ok"),
            ) -> dict[str, object]:
                _ = cls
                return {"dep": dep, "label": label}

    container = ServiceContainer()
    await register_services_from_registry(container, registry=registry)
    alpha = cast(dict[str, object], await container.aget_by_key("alpha_consumer"))
    beta = cast(dict[str, object], await container.aget_by_key("beta_consumer"))

    alpha_dep = cast(dict[str, str], alpha["dep"])
    beta_dep = cast(dict[str, str], beta["dep"])
    assert alpha_dep["name"] == "dep-a"
    assert beta_dep["name"] == "dep-b"
    assert alpha["label"] == "alpha-ok"
    assert beta["label"] == "beta-ok"


@pytest.mark.asyncio
async def test_singleton_missing_required_keyword_only_param_fails_during_registration() -> None:
    registry = AppServiceRegistry()
    with _capture_here(registry):
        @Service("bad_singleton_kw")
        class BadSingletonKeywordOnly(BaseService):
            @classmethod
            async def create(cls, *, config: str) -> BadSingletonKeywordOnly:
                _ = config
                return cls()

    container = ServiceContainer()
    with pytest.raises(
        TypeError,
        match="Invalid singleton service 'bad_singleton_kw'.*missing required keyword-only argument 'config'",
    ):
        await register_services_from_registry(container, registry=registry)


@pytest.mark.asyncio
async def test_transient_positional_args_do_not_override_require_default() -> None:
    registry = AppServiceRegistry()
    with _capture_here(registry):
        @Service("dep")
        class DepService(BaseService):
            @classmethod
            async def create(cls) -> str:
                _ = cls
                return "dep"

        @Service("collector", lifetime=ServiceLifetime.TRANSIENT)
        class CollectorService(BaseService):
            @classmethod
            async def create(
                cls,
                dep=Require("dep"),
                *args: object,
            ) -> tuple[str, tuple[object, ...]]:
                _ = cls
                return cast(str, dep), args

    container = ServiceContainer()
    await register_services_from_registry(container, registry=registry)
    payload = await container.aget_by_key("collector", 1, 2)
    assert payload == ("dep", (1, 2))
