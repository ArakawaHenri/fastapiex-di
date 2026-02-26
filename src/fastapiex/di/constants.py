from __future__ import annotations

# app.state attribute names
APP_STATE_SC_REGISTRY_ATTR = "sc_registry"
APP_STATE_DI_CONFIG_ATTR = "di_config"
APP_STATE_DI_SERVICE_REGISTRY_ATTR = "di_service_registry"
APP_STATE_DI_REGISTERED_SERVICE_ORIGINS_ATTR = "di_registered_service_origins"
APP_STATE_DI_GLOBAL_REFRESH_LOCK_ATTR = "di_global_refresh_lock"

# request/registry internals
REQUEST_FAILED_STATE_KEY = "_svc_request_failed"
SERVICE_DEFAULT_DESTROY_MARKER = "__service_default_destroy_noop__"
SERVICE_DEFINITION_ATTR = "__fastapi_di_definition__"
