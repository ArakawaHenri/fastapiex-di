# fastapiex-di

Production-ready FastAPI extension for service registry and dependency injection.

## Installation

```bash
uv add fastapiex-di
```

## Quick Start

```python
from fastapi import FastAPI
from fastapiex.di import BaseService, Inject, Service, install_di

app = FastAPI()
install_di(
    app,
    service_packages=["myapp.services"],
)


@Service("clock_service", eager=True)
class ClockService(BaseService):
    @classmethod
    async def create(cls) -> "ClockService":
        return cls()


@app.get("/health")
async def health(svc: ClockService = Inject("clock_service")):
    return {"ok": isinstance(svc, ClockService)}
```

`install_di(...)` wires startup/shutdown lifecycle automatically.

Important:

- Keep service modules lazily imported by `install_di(...)` package scanning.
- Avoid importing decorated service modules before `install_di(...)` runs.

## Quickstart

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

`@Service` and `@ServiceDict` run at import time.
The registration capture window is opened during `install_di(...)` startup import scanning.
If decorated services are imported before that window, startup fails with:
`No active app service registry capture`.

Use a separate service module (for example `demo/services.py`) and point
`service_packages` to that module path.

## Import Timing Rules

| Do | Don't |
| --- | --- |
| `install_di(app, service_packages=["demo.services"])` | Put `@Service` classes in `main.py` and import them before startup |
| Keep decorated services under a dedicated module/package | `from demo.services import PingService` in `demo/main.py` |
| Let DI scan import service modules during startup | Manually import decorated service modules in app bootstrap |

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

- Keep all `@Service` / `@ServiceDict` classes under one or more explicit packages (for example `app.services`).
- Keep route handlers under `app.api.*`, and resolve dependencies via `Inject(...)` only.
- Keep framework config (`settings`, logging, middleware wiring) under `app.core.*`.

## Service Registration

### 1. Named service

```python
from fastapiex.di import BaseService, Service


@Service("user_service")
class UserService(BaseService):
    @classmethod
    async def create(cls) -> "UserService":
        return cls()
```

### 2. Anonymous service (type-only)

```python
from fastapiex.di import BaseService, Service


class UserCache:
    pass


@Service
class UserCacheProvider(BaseService):
    @classmethod
    async def create(cls) -> UserCache:
        return UserCache()
```

### 3. ServiceDict expansion

```python
from fastapiex.di import BaseService, ServiceDict


@ServiceDict("{}_db_service", dict={"main": {"dsn": "sqlite+aiosqlite:///main.db"}})
class DatabaseService(BaseService):
    @classmethod
    async def create(cls, dsn: str) -> "DatabaseService":
        instance = cls()
        instance.dsn = dsn
        return instance
```

## Declaring Service-to-Service Dependencies

Use `require(...)` in `create(...)` defaults.

```python
from fastapiex.di import BaseService, Service, require


@Service("repo_service")
class RepoService(BaseService):
    @classmethod
    async def create(cls, db=require("main_db_service")) -> "RepoService":
        _ = db
        return cls()
```

You can also depend on type:

```python
@Service("consumer")
class ConsumerService(BaseService):
    @classmethod
    async def create(cls, cache=require(UserCache)) -> "ConsumerService":
        _ = cache
        return cls()
```

## Injecting Services in FastAPI Endpoints

### Key-based

```python
from fastapiex.di import Inject


@app.get("/users")
async def list_users(repo=Inject("repo_service")):
    return {"ok": True}
```

### Type-based (only when exactly one provider exists)

```python
@app.get("/cache")
async def cache_state(cache: UserCache = Inject(UserCache)):
    return {"ok": isinstance(cache, UserCache)}
```

## Production Settings

`install_di(...)` options:

- `service_packages`: package(s) to scan for decorated services.
- `strict` (default `True`): fail startup on DI/registry errors.
- `allow_private_modules` (default `False`): include modules with underscore segments.
- `auto_add_finalizer_middleware` (default `True`): auto install transient cleanup middleware.
- `freeze_container_after_startup` (default `True`): block runtime service registrations.
- `freeze_service_registry_after_startup` (default `False`): freeze this app's scoped service registry after startup.
- `unfreeze_service_registry_on_shutdown` (default `True`): unfreeze this app's registry when app exits.
- `eager_init_timeout_sec` (optional): timeout for eager singleton initialization.

Recommended production defaults:

```python
install_di(
    app,
    service_packages=["myapp.services"],
    strict=True,
    freeze_container_after_startup=True,
    freeze_service_registry_after_startup=True,
    eager_init_timeout_sec=30,
)
```

## Safety and Worker Model

- Container enforces single event-loop usage.
- Container rejects cross-process reuse.
- Registry maps container by current process/thread/event-loop context.
- Runtime service registry is app-scoped, so freeze/unfreeze does not leak across apps.
- Transient finalizers run after request completion.
- Transient finalizers also run after WebSocket connection teardown.
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

- `Duplicate service registration for key`: same key registered more than once.
- `No service registered for type`: missing provider for type-based injection.
- `Multiple services registered for type`: use key-based injection instead.
- `Detected circular service dependency`: dependency graph has a cycle.
- `Cannot register services after container registrations are frozen`: runtime registration attempted after startup.
- `No active app service registry capture`: decorated service module was imported before `install_di(...)` startup import scanning.
  Fix:
  1. Move `@Service`/`@ServiceDict` classes into a dedicated module (for example `demo/services.py`).
  2. Set `install_di(..., service_packages=["demo.services"])` to that module path.
  3. Remove early imports of that service module from `main.py`.

## Public API

```python
from fastapiex.di import (
    AppServiceRegistry,
    BaseService,
    Inject,
    Service,
    ServiceDict,
    ServiceContainer,
    ServiceLifetime,
    capture_service_registrations,
    install_di,
    require,
)
```
