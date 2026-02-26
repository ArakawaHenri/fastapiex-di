from __future__ import annotations

from fastapiex.di import BaseService, Service


@Service("external_late_global_service")
class ExternalLateGlobalService(BaseService):
    @classmethod
    async def create(cls) -> dict[str, str]:
        _ = cls
        return {"scope": "late-global"}

