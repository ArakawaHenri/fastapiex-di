from __future__ import annotations

from fastapiex.di import BaseService, Service


@Service("external_global_service")
class ExternalGlobalService(BaseService):
    @classmethod
    async def create(cls) -> dict[str, str]:
        _ = cls
        return {"scope": "global"}

