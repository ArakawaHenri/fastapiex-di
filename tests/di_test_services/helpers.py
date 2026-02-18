from __future__ import annotations

_ORDER_SINK: list[str] | None = None
_ORDER_SEQ = 0


def set_order_sink(sink: list[str] | None) -> None:
    global _ORDER_SINK, _ORDER_SEQ
    _ORDER_SINK = sink
    _ORDER_SEQ = 0


def append_order(value: str) -> None:
    if _ORDER_SINK is not None:
        _ORDER_SINK.append(value)


def next_order_seq() -> int:
    global _ORDER_SEQ
    _ORDER_SEQ += 1
    return _ORDER_SEQ
