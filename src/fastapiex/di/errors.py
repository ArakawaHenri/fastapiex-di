from __future__ import annotations


class DIError(Exception):
    """Base class for all package-defined DI exceptions."""


class DITypeError(DIError, TypeError):
    """Base class for DI API/type contract violations."""


class DIValueError(DIError, ValueError):
    """Base class for DI value/domain validation failures."""


class DIRuntimeError(DIError, RuntimeError):
    """Base class for DI runtime failures."""


class DIConfigurationError(DIValueError):
    """Raised when DI configuration values are invalid."""


class DIAlreadyInstalledError(DIRuntimeError):
    """Raised when install_di() is invoked more than once for the same app."""


class ServiceContainerError(DIRuntimeError):
    """Base class for container lifecycle/access/runtime errors."""


class ServiceContainerAccessError(ServiceContainerError):
    """Raised when the container is accessed from an invalid context."""


class ServiceContainerStateError(ServiceContainerError):
    """Raised when the container is in an invalid internal state."""


class ServiceRegistrationError(ServiceContainerError):
    """Raised when service registration invariants are violated."""


class DuplicateServiceRegistrationError(ServiceRegistrationError):
    """Raised when a key is registered more than once."""


class InvalidServiceDefinitionError(DITypeError):
    """Raised when service ctor/dtor/lifetime definitions are invalid."""


class ServiceFactoryContractError(ServiceContainerError):
    """Raised when factory generator/contextmanager contracts are violated."""


class ServiceResolutionError(ServiceContainerError):
    """Base class for resolution-time service lookup failures."""


class UnregisteredServiceError(ServiceResolutionError):
    """Base class for missing service registrations."""


class UnregisteredServiceByKeyError(UnregisteredServiceError):
    """Raised when resolving by key and no service is registered."""


class UnregisteredServiceByTypeError(UnregisteredServiceError):
    """Raised when resolving by type and no service is registered."""


class AmbiguousServiceByTypeError(ServiceResolutionError):
    """Raised when type-based resolution has multiple matching services."""


class ServiceRegistryError(DIRuntimeError):
    """Base class for registry/planning errors."""


class ServiceDependencyError(ServiceRegistryError):
    """Raised when dependency graph constraints are violated."""


class CircularServiceDependencyError(ServiceDependencyError):
    """Raised when a circular dependency is detected."""


class ServiceMappingError(DITypeError):
    """Raised when ServiceMap mappings cannot be safely composed."""
