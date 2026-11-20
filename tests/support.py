from typing import Any, Iterable


def pluck(items: Iterable[Any], key: str) -> list[Any]:
    values: list[Any] = []
    for item in items:
        values.append(item[key])
    return values


def index_by(items: Iterable[Any], key: str) -> dict[Any, Any]:
    indexed: dict[Any, Any] = {}
    for item in items:
        indexed[item[key]] = item
    return indexed


def pluck_optional(items: Iterable[Any], key: str) -> list[Any]:
    values: list[Any] = []
    for item in items:
        values.append(item.get(key))
    return values
