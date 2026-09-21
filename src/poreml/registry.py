"""Name-to-object registries — the extension seams for tasks, models, descriptors, and errors."""

from collections.abc import Callable


class Registry[T]:
    """A name-to-object map populated by decorator.

    Registration is strict: a duplicate name raises rather than overwriting, because a
    benchmark where two implementations answer to one name reports the wrong number.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        if name not in self._items:
            known = ", ".join(self.names()) or "<none>"
            raise KeyError(f"unknown {self.kind} {name!r}; registered: {known}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __repr__(self) -> str:
        return f"Registry(kind={self.kind!r}, names={self.names()})"


TASKS: Registry = Registry("task")
MODELS: Registry = Registry("model")
DESCRIPTORS: Registry = Registry("descriptor")
ERRORS: Registry = Registry("error")
