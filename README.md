# fastapiex-di

DI extension for FastAPIEx: Production-ready FastAPI extension for service registry and dependency injection.

## Installation

```bash
uv add fastapiex-di
```

## Quick Start

Use this exact structure from your project root:

```text
base_dir/
├── .venv/
└── demo/
    ├── __init__.py
    ├── main.py
    └── services.py
```

`demo/services.py`:

```python
from fastapiex.di import BaseService, Service


@Service("ping_service")
class PingService(BaseService):
    @classmethod
    async def create(cls) -> "PingService":
        return cls()

    async def ping(self) -> str:
        return "pong"
```

`demo/main.py`:

```python
from fastapi import FastAPI
from fastapiex.di import Inject, install_di

app = FastAPI()
install_di(app, service_packages=["demo.services"])


@app.get("/ping")
async def ping(svc=Inject("ping_service")):
    return {"msg": await svc.ping()}
```

Run:

```bash
uv run uvicorn demo.main:app --reload
```

Then open `http://127.0.0.1:8000/ping` and expect:

```json
{"msg":"pong"}
```

Notes:

- `service_packages=["demo.services"]` must be a real import path, not a filesystem path.
- Do not import `demo.services` in `demo/main.py`; let `install_di(...)` import it during startup.

## Why Not Single-File

`@Service` and `@ServiceMap` run at import time.
`install_di(...)` discovers definitions by importing and scanning modules under `service_packages`.

Early imports are allowed, but service modules should still live in a dedicated package
and be included in `service_packages`. This keeps startup wiring predictable and avoids
bootstrap-time circular imports.

## Import Timing Rules

| Do | Don't |
| --- | --- |
| `install_di(app, service_packages=["demo.services"])` | Rely on services outside `service_packages` unless `use_global_service_registry=True` is intentional |
| Keep decorated services under a dedicated module/package | Scatter `@Service` classes across app bootstrap and route modules |
| Let DI own service-module discovery during startup | Build side-effect-heavy bootstrap imports that increase cycle risk |

## Project Layout Contract

`service_packages` accepts Python import paths, not filesystem paths.

Examples:

- Valid: `service_packages=["demo.services"]`
- Valid: `service_packages=["myapp.services"]`
- Invalid: `service_packages=["demo/services.py"]`
- Invalid: `service_packages=["./demo/services"]`

## Ideal App Layout

Example project structure that keeps DI wiring predictable:

```text
myapp/
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── settings.py
│   │   └── logging.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── users.py
│   └── services/
│       ├── __init__.py
│       ├── database.py
│       ├── cache.py
│       └── user_repo.py
└── pyproject.toml
```

`app/main.py`:

```python
from fastapi import FastAPI
from fastapiex.di import install_di

app = FastAPI()
install_di(app, service_packages=["app.services"])
```

Guidelines:

- Keep all `@Service` / `@ServiceMap` classes under one or more explicit packages (for example `app.services`).
- Keep route handlers under `app.api.*`, and resolve dependencies via `Inject(...)` only.
- Keep framework config (settings, logging, middleware wiring) under `app.core.*`.

## Service Registration

### Naming Conventions (Recommended)

- Singleton services: use `Service` suffix (for example `UserRepoService`).
- Transient services: use `ServiceT` suffix (for example `UserRepoServiceT`).
- Generator/contextmanager-style services: use `ServiceG` suffix (for example `UserRepoServiceG`).

### 1. Singleton + eager

```python
from fastapiex.di import BaseService, Service


@Service("app_config_service", eager=True)
class AppConfigService(BaseService):
    @classmethod
    async def create(cls) -> "AppConfigService":
        return cls()
```

`eager=True` only applies to singleton services. Transient services cannot be eager.

### 2. Transient service

```python
from fastapiex.di import BaseService, Service


@Service("request_context_service_t", lifetime="transient")
class RequestContextServiceT(BaseService):
    @classmethod
    async def create(cls) -> "RequestContextServiceT":
        return cls()
```

### 3. `exposed_type` for type-based resolution

```python
from typing import Protocol

from fastapiex.di import BaseService, Service


class UserRepo(Protocol):
    async def list_users(self) -> list[str]:
        ...


@Service("repo_service", exposed_type=UserRepo)
class UserRepoService(BaseService):
    @classmethod
    async def create(cls) -> "UserRepoService":
        return cls()

    async def list_users(self) -> list[str]:
        return ["alice", "bob"]
```

### 4. Anonymous service (type-only)

```python
from fastapiex.di import BaseService, Service


class UserCache:
    pass


@Service
class UserCacheService(BaseService):
    @classmethod
    async def create(cls) -> UserCache:
        return UserCache()
```

### 5. ServiceMap expansion

```python
from fastapiex.di import BaseService, ServiceMap


@ServiceMap("{}_db_service", mapping={"main": {"dsn": "sqlite+aiosqlite:///main.db"}})
class DatabaseService(BaseService):
    @classmethod
    async def create(cls, dsn: str) -> "DatabaseService":
        instance = cls()
        instance.dsn = dsn
        return instance
```

## Declaring Service-to-Service Dependencies

Use `Require(...)` in `create(...)` defaults.
The example below reuses `UserRepo` and `UserCache` defined above.

```python
from fastapiex.di import BaseService, Service, Require


@Service("user_query_service_t", lifetime="transient")
class UserQueryServiceT(BaseService):
    @classmethod
    async def create(
        cls,
        repo=Require(UserRepo),
        cache=Require(UserCache),
    ) -> "UserQueryServiceT":
        _ = repo, cache
        return cls()
```

## Injecting Services in FastAPI Endpoints

`Inject(...)` requires an explicit target. Pass either a registered service key
or a service type. `Inject()` with no arguments is invalid.

### Key-based

```python
from fastapiex.di import Inject


@app.get("/users/by-key")
async def users_by_key(repo=Inject("repo_service")):
    return {"users": await repo.list_users()}
```

### Type-based (only when exactly one provider exists)

```python
@app.get("/users/by-type")
async def users_by_type(repo: UserRepo = Inject(UserRepo)):
    return {"users": await repo.list_users()}
```

### Nested

```python
@app.get("/nested")
async def nested(
    query_service: UserQueryServiceT = Inject(
        "user_query_service_t",
        repo=Inject("repo_service"),
        cache=Inject(UserCache),
    ),
):
    return {"ok": isinstance(query_service, UserQueryServiceT)}
```

## Production Settings

`install_di(...)` options:

- `service_packages`: package(s) to scan for decorated services.
- `strict` (default `True`): fail startup on DI/registry errors.
- `use_global_service_registry` (default `False`): maintain a global registry for services declared outside all configured `service_packages`.
  On unresolved injection, DI always performs one app-local refresh attempt; with this flag enabled, that refresh also merges global definitions.
- `allow_private_modules` (default `False`): include underscore modules during package scan/import.
  Private modules imported manually at runtime can still register if they belong to configured `service_packages`.
- `eager_init_timeout_sec` (optional): timeout for eager singleton initialization.

Recommended production defaults:

```python
install_di(
    app,
    service_packages=["myapp.services"],
    strict=True,
    eager_init_timeout_sec=30,
)
```

## Safety and Worker Model

- Container enforces single event-loop usage.
- Container rejects cross-process reuse.
- Registry maps container by current process/thread/event-loop context.
- Cleanup-requiring transient services must resolve inside an active FastAPI request/WebSocket scope.
- Transient generator exit runs in a DI-managed function scope (transaction semantics).
- Transient `destroy()` callbacks run in a DI-managed request scope after response/background completion.
- Singleton teardown runs on shutdown in reverse order.

## Supply-Chain Security

Install security tooling group:

```bash
uv sync --locked --no-default-groups --group security
```

Run checks:

```bash
./scripts/supply_chain_check.sh
```

## Common Errors

- DI-defined API/runtime exceptions are custom (`fastapiex.di.errors.*`) and inherit `DIError`.
  Import-time failures from Python (for example `ModuleNotFoundError` from invalid `service_packages`) can still bubble up.
- `Duplicate service registration for key`: same key registered more than once.
- `No service registered for type`: missing provider for type-based injection.
- `Multiple services registered for type`: use key-based injection instead.
- `Detected circular service dependency`: dependency graph has a cycle.
- `Service container not initialized for the current event loop`: DI startup did not complete for this app loop.
  Fix:
  1. Ensure `install_di(...)` is called exactly once during app creation.
  2. Confirm `service_packages` uses import paths that can be imported at startup.
  3. Keep `strict=True` in production so startup fails fast on DI wiring errors.

## Public API

```python
from fastapiex.di import (
    BaseService,
    Inject,
    Require,
    Service,
    ServiceMap,
    ServiceLifetime,
    install_di,
)
```

Advanced APIs are available from submodules:

```python
from fastapiex.di.container import ServiceContainer
from fastapiex.di.registry import (
    AppServiceRegistry,
    capture_service_registrations,
)
```
