from __future__ import annotations

from fastapiex.di import BaseService, Service


@Service("late_private_service")
class LatePrivateService(BaseService):
    @classmethod
    async def create(cls) -> dict[str, str]:
        _ = cls
        return {"scope": "private"}

