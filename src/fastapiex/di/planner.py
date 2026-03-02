from __future__ import annotations

import inspect
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from .container import ServiceContainer
from .exceptions import (
    CircularServiceDependencyError,
    InvalidServiceDefinitionError,
    ServiceDependencyError,
    ServiceMappingError,
    ServiceRegistryError,
)
from .registry import (
    AppServiceRegistry,
    ExpandedDefinition,
    RequiredDependency,
    ResolvedByInternalId,
    ResolvedSpec,
    _CompiledService,
    _is_origin_included,
    _ResolvedDependency,
)
from .types import ServiceLifetime


def _render_service_key(key_template: str, dict_key: str) -> str:
    if "{}" in key_template:
        return key_template.replace("{}", dict_key)
    return f"{dict_key}_{key_template}"


def _coerce_mapping_value(
    value: object,
    *,
    signature: inspect.Signature,
    dependency_params: set[str],
) -> tuple[tuple[object, ...], dict[str, object]]:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        raw = value.model_dump()
        if not isinstance(raw, Mapping):
            raise ServiceMappingError("model_dump() must return a mapping")
        mapping_kwargs = dict(raw)
        _validate_mapping_kwargs(signature=signature, mapping_kwargs=mapping_kwargs)
        return (), mapping_kwargs

    if isinstance(value, Mapping):
        mapping_kwargs = dict(value)
        _validate_mapping_kwargs(signature=signature, mapping_kwargs=mapping_kwargs)
        return (), mapping_kwargs

    # For scalar shorthand values we only accept deterministic bindings:
    # 1) Prefer keyword-capable parameters to avoid positional drift.
    # 2) Fallback to positional-only only when it is the first positional slot.
    keyword_target: str | None = None
    for name, parameter in signature.parameters.items():
        if name in dependency_params:
            continue
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            keyword_target = name
            break
    if keyword_target is not None:
        return (), {keyword_target: value}

    positional_target: inspect.Parameter | None = None
    for name, parameter in signature.parameters.items():
        if name in dependency_params:
            continue
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            positional_target = parameter
            break
    if positional_target is None:
        raise ServiceMappingError("Unable to map ServiceMap value to ctor parameters")

    positional_slots = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    target_index = next(
        index
        for index, parameter in enumerate(positional_slots)
        if parameter.name == positional_target.name
    )
    if target_index != 0:
        raise ServiceMappingError(
            "Scalar ServiceMap value cannot bind positional-only parameter "
            f"'{positional_target.name}' because earlier positional parameters exist. "
            "Use mapping={...} for explicit binding."
        )
    return (value,), {}


def _validate_mapping_kwargs(
    *,
    signature: inspect.Signature,
    mapping_kwargs: Mapping[str, object],
) -> None:
    positional_only = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY
    }
    for name in mapping_kwargs:
        if name in positional_only:
            raise ServiceMappingError(
                "ServiceMap mapping cannot pass positional-only "
                f"parameter '{name}' by key."
            )


def _resolve_source(
    source: Mapping[str, object] | Callable[[], Mapping[str, object]],
) -> Mapping[str, object]:
    resolved = source() if callable(source) else source
    if not isinstance(resolved, Mapping):
        raise ServiceMappingError("ServiceMap source must resolve to a mapping")
    return resolved


def _expand_definitions(
    *,
    registry: AppServiceRegistry,
    include_packages: tuple[str, ...] | None = None,
) -> list[ExpandedDefinition]:
    expanded: list[ExpandedDefinition] = []

    for definition in registry.definitions():
        if include_packages is not None and not _is_origin_included(
            definition.origin,
            include_packages,
        ):
            continue
        signature = inspect.signature(definition.ctor)
        dependency_params = set(definition.dependencies.keys())

        if definition.source is None:
            expanded.append(
                (
                    definition,
                    definition.key_template,
                    (),
                    {},
                    definition.dependencies,
                    signature,
                )
            )
            continue

        if definition.key_template is None:
            raise ServiceRegistryError(
                f"Service definition '{definition.origin}' has dict source but no key template."
            )

        source_mapping = _resolve_source(definition.source)
        for raw_dict_key, raw_value in source_mapping.items():
            dict_key = str(raw_dict_key)
            service_key = _render_service_key(definition.key_template, dict_key)
            static_args, static_kwargs = _coerce_mapping_value(
                raw_value,
                signature=signature,
                dependency_params=dependency_params,
            )
            static_bound: inspect.BoundArguments
            try:
                static_bound = signature.bind_partial(*static_args, **static_kwargs)
            except TypeError as exc:
                raise ServiceMappingError(
                    f"Invalid ServiceMap mapping value for key '{service_key}' in "
                    f"'{definition.origin}': {exc}"
                ) from exc
            overridden_params = set(static_bound.arguments.keys())
            deps = {
                param_name: dep.render_for_dict_key(dict_key)
                for param_name, dep in definition.dependencies.items()
                if param_name not in overridden_params
            }
            expanded.append(
                (
                    definition,
                    service_key,
                    static_args,
                    static_kwargs,
                    deps,
                    signature,
                )
            )

    return expanded


def _resolve_dependency_targets(
    compiled: list[ExpandedDefinition],
) -> list[_CompiledService]:
    by_internal_id: dict[str, ResolvedByInternalId] = {}
    public_key_index: dict[str, str] = {}
    type_index: dict[object, list[str]] = defaultdict(list)

    anon_seq = 0
    for definition, key, static_args, static_kwargs, deps, signature in compiled:
        if key is not None:
            existing_internal_id = public_key_index.get(key)
            if existing_internal_id is not None:
                first = by_internal_id[existing_internal_id][0]
                raise ServiceRegistryError(
                    f"Duplicate service key '{key}' from {first.origin} and {definition.origin}."
                )
            internal_id = key
            public_key_index[key] = internal_id
        else:
            anon_seq += 1
            internal_id = f"anonymous:{definition.origin}:{anon_seq}"

        service_type = definition.exposed_type
        if service_type is None:
            service_type = ServiceContainer._infer_service_type(definition.ctor)
        if service_type is None:
            service_type = definition.service_cls
        by_internal_id[internal_id] = (
            definition,
            deps,
            signature,
            static_args,
            static_kwargs,
            service_type,
            key,
        )
        type_index[service_type].append(internal_id)

    dependency_map: dict[str, set[str]] = {}
    resolved: dict[str, ResolvedSpec] = {}

    def _service_label(internal_id: str) -> str:
        definition, _, _, _, _, _, registration_key = by_internal_id[internal_id]
        return registration_key or definition.origin

    for internal_id, (
        definition,
        deps,
        signature,
        static_args,
        static_kwargs,
        _service_type,
        registration_key,
    ) in by_internal_id.items():
        service_label = registration_key or definition.origin
        dep_internal_ids: set[str] = set()
        resolved_deps: list[_ResolvedDependency] = []

        for param_name, dep in deps.items():
            if isinstance(dep.target, str):
                dep_key = dep.target
                dep_internal_id = public_key_index.get(dep_key)
                if dep_internal_id is None:
                    raise ServiceDependencyError(
                        f"Service '{service_label}' depends on '{dep_key}', but that service is not registered."
                    )
                resolved_deps.append(
                    _ResolvedDependency(
                        param_name=param_name,
                        required=dep,
                    )
                )
            else:
                candidate_plan_keys = type_index.get(dep.target, [])
                if not candidate_plan_keys:
                    raise ServiceDependencyError(
                        f"Service '{service_label}' depends on type {dep.target!r}, but no service provides that type."
                    )
                if len(candidate_plan_keys) > 1:
                    candidate_labels = [_service_label(candidate_id) for candidate_id in candidate_plan_keys]
                    raise ServiceDependencyError(
                        "Service "
                        f"'{service_label}' depends on type {dep.target!r}, "
                        f"but multiple services provide it: {candidate_labels}."
                    )
                dep_internal_id = candidate_plan_keys[0]
                resolved_deps.append(
                    _ResolvedDependency(
                        param_name=param_name,
                        required=dep,
                    )
                )

            target_definition = by_internal_id[dep_internal_id][0]
            target_key = by_internal_id[dep_internal_id][6]
            target_label = target_key or target_definition.origin
            if (
                definition.lifetime == ServiceLifetime.SINGLETON
                and target_definition.lifetime == ServiceLifetime.TRANSIENT
                and not dep.allow_transient
            ):
                raise ServiceDependencyError(
                    f"Singleton service '{service_label}' depends on transient service '{target_label}'. "
                    "Use Require(..., allow_transient=True) only if this is intentional."
                )

            dep_internal_ids.add(dep_internal_id)

        dependency_map[internal_id] = dep_internal_ids
        resolved[internal_id] = (
            definition,
            signature,
            static_args,
            static_kwargs,
            tuple(resolved_deps),
            by_internal_id[internal_id][5],
            registration_key,
        )

    registration_order = _topological_registration_order(dependency_map)

    result: list[_CompiledService] = []
    for internal_id in registration_order:
        (
            definition,
            signature,
            static_args,
            static_kwargs,
            resolved_dependencies,
            service_type,
            registration_key,
        ) = resolved[internal_id]
        result.append(
            _CompiledService(
                origin=definition.origin,
                internal_id=internal_id,
                key=registration_key,
                lifetime=definition.lifetime,
                eager=definition.eager,
                ctor=definition.ctor,
                dtor=definition.dtor,
                signature=signature,
                static_args=static_args,
                static_kwargs=static_kwargs,
                dependencies=resolved_dependencies,
                service_type=service_type,
            )
        )

    return result


def _topological_registration_order(dependency_map: dict[str, set[str]]) -> list[str]:
    indegree = dict.fromkeys(dependency_map, 0)
    outgoing: dict[str, set[str]] = defaultdict(set)

    for node, deps in dependency_map.items():
        indegree[node] = len(deps)
        for dep in deps:
            outgoing[dep].add(node)

    queue = deque(node for node, degree in indegree.items() if degree == 0)
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for successor in outgoing.get(node, set()):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)

    if len(order) != len(dependency_map):
        cycle = _detect_cycle(dependency_map)
        if cycle:
            cycle_path = " -> ".join(cycle)
            raise CircularServiceDependencyError(
                f"Detected circular service dependency: {cycle_path}"
            )
        raise CircularServiceDependencyError("Detected circular service dependency")

    return order


def _detect_cycle(dependency_map: dict[str, set[str]]) -> list[str] | None:
    UNVISITED = 0
    VISITING = 1
    VISITED = 2

    state: dict[str, int] = dict.fromkeys(dependency_map, UNVISITED)
    stack: list[str] = []

    def dfs(node: str) -> list[str] | None:
        state[node] = VISITING
        stack.append(node)

        for dep in dependency_map.get(node, set()):
            dep_state = state.get(dep, UNVISITED)
            if dep_state == UNVISITED:
                cycle = dfs(dep)
                if cycle is not None:
                    return cycle
            elif dep_state == VISITING:
                start = stack.index(dep)
                return stack[start:] + [dep]

        stack.pop()
        state[node] = VISITED
        return None

    for node in dependency_map:
        if state[node] == UNVISITED:
            cycle = dfs(node)
            if cycle is not None:
                return cycle

    return None


def build_service_plan(
    *,
    registry: AppServiceRegistry,
    include_packages: Sequence[str] | None = None,
) -> list[_CompiledService]:
    include = tuple(include_packages) if include_packages is not None else None
    expanded = _expand_definitions(registry=registry, include_packages=include)
    compiled = _resolve_dependency_targets(expanded)
    _validate_singleton_specs(compiled)
    return compiled


def _validate_singleton_specs(specs: Sequence[_CompiledService]) -> None:
    for spec in specs:
        if spec.lifetime != ServiceLifetime.SINGLETON:
            continue
        dependency_defaults = {
            dependency.param_name: dependency.required
            for dependency in spec.dependencies
        }
        service_label = spec.key or spec.origin
        try:
            _compose_ctor_call_arguments(
                signature=spec.signature,
                dependency_defaults=dependency_defaults,
                static_args=spec.static_args,
                static_kwargs=spec.static_kwargs,
                runtime_args=(),
                runtime_kwargs={},
                service_label=service_label,
            )
        except TypeError as exc:
            raise InvalidServiceDefinitionError(
                f"Invalid singleton service '{service_label}': {exc}"
            ) from exc


def _compose_ctor_call_arguments(
    *,
    signature: inspect.Signature,
    dependency_defaults: Mapping[str, RequiredDependency],
    static_args: tuple[object, ...],
    static_kwargs: Mapping[str, object],
    runtime_args: tuple[object, ...],
    runtime_kwargs: Mapping[str, object],
    service_label: str,
) -> tuple[tuple[object, ...], dict[str, object]]:
    parameter_map = signature.parameters
    positional_params: list[inspect.Parameter] = []
    kw_only_params: list[inspect.Parameter] = []
    has_var_keyword = False
    has_var_positional = False

    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_params.append(parameter)
        elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            kw_only_params.append(parameter)
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True

    base_bound = signature.bind_partial(*static_args, **dict(static_kwargs))

    values: dict[str, object] = {}
    var_positional_values: list[object] = []
    var_keyword_values: dict[str, object] = {}
    for name, value in base_bound.arguments.items():
        parameter = parameter_map[name]
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            var_positional_values.extend(cast(tuple[object, ...], value))
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            var_keyword_values.update(cast(dict[str, object], value))
            continue
        values[name] = value

    unresolved_required_positional = [
        parameter.name
        for parameter in positional_params
        if parameter.name not in values
        and parameter.name not in dependency_defaults
        and parameter.default is inspect.Parameter.empty
    ]

    runtime_positional_values = list(runtime_args)
    runtime_positional_assigned: set[str] = set()
    for param_name in unresolved_required_positional:
        if not runtime_positional_values:
            break
        values[param_name] = runtime_positional_values.pop(0)
        runtime_positional_assigned.add(param_name)

    if runtime_positional_values:
        if not has_var_positional:
            raise InvalidServiceDefinitionError(
                f"Service '{service_label}' received too many positional arguments."
            )
        var_positional_values.extend(runtime_positional_values)

    for key, value in runtime_kwargs.items():
        runtime_param = parameter_map.get(key)
        if runtime_param is None:
            if not has_var_keyword:
                raise InvalidServiceDefinitionError(
                    f"Service '{service_label}' got an unexpected keyword argument '{key}'."
                )
            var_keyword_values[key] = value
            continue
        if runtime_param.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise InvalidServiceDefinitionError(
                f"Service '{service_label}' positional-only parameter '{key}' passed as keyword."
            )
        if runtime_param.kind == inspect.Parameter.VAR_POSITIONAL:
            raise InvalidServiceDefinitionError(
                f"Service '{service_label}' invalid keyword argument for *{key}."
            )
        if key in runtime_positional_assigned:
            raise InvalidServiceDefinitionError(
                f"Service '{service_label}' got multiple values for argument '{key}'."
            )
        if runtime_param.kind == inspect.Parameter.VAR_KEYWORD:
            var_keyword_values[key] = value
            continue
        values[key] = value

    for parameter in (*positional_params, *kw_only_params):
        if parameter.name in values:
            continue
        dependency_default = dependency_defaults.get(parameter.name)
        if dependency_default is not None:
            values[parameter.name] = dependency_default
            continue
        if parameter.default is not inspect.Parameter.empty:
            values[parameter.name] = parameter.default
            continue
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            raise InvalidServiceDefinitionError(
                f"Service '{service_label}' missing required keyword-only argument '{parameter.name}'."
            )
        raise InvalidServiceDefinitionError(
            f"Service '{service_label}' missing required argument '{parameter.name}'."
        )

    final_args = [values[parameter.name] for parameter in positional_params]
    if has_var_positional:
        final_args.extend(var_positional_values)
    final_kwargs = {
        parameter.name: values[parameter.name]
        for parameter in kw_only_params
        if parameter.name in values
    }
    if has_var_keyword:
        final_kwargs.update(var_keyword_values)

    return tuple(final_args), final_kwargs
