from contextvars import ContextVar
from functools import wraps

BUILD_CONTEXT = ContextVar("BUILD_CONTEXT")
BUILD_CONTEXT.set(False)


def allow_build(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        BUILD_CONTEXT.set(True)

        if type(func) == classmethod:
            obj = func.__func__(*args, **kwargs)
        elif callable(func):
            obj = func(*args, **kwargs)
        else:
            raise TypeError(func)

        BUILD_CONTEXT.set(False)
        return obj

    return wrap


# from contextvars import Context

# T = TypeVar("T")


# class NoPublicConstructor(type):
#     """Metaclass that ensures a private constructor

#     If a class uses this metaclass like this:

#         class SomeClass(metaclass=NoPublicConstructor):
#             pass

#     If you try to instantiate your class (`SomeClass()`),
#     a `TypeError` will be thrown.
#     """

#     def __call__(cls, *args, **kwargs):
#         raise TypeError(
#             f"{cls.__module__}.{cls.__qualname__} has no public constructor"
#         )

#     def _create(cls: Type[T], *args: Any, **kwargs: Any) -> T:
#         return super().__call__(*args, **kwargs)  # type: ignore
