# methodize.py
import inspect
import types
from functools import wraps


def _strip_first_param_from_doc(doc: str) -> str:
    """Remove the first parameter entry from a NumPy-style docstring.

    Best-effort removal that only drops the first parameter when its type is
    ``LineageTree``. Docstrings that do not match the expected NumPy layout are
    returned unchanged.

    Parameters
    ----------
    doc : str
        The docstring to process.

    Returns
    -------
    str
        The docstring with the leading ``lT : LineageTree`` parameter removed,
        or the original docstring when no matching entry is found.
    """
    if not doc:
        return doc

    lines = doc.splitlines()
    # Look for a "Parameters" section, then entries shaped like:
    #   name : type
    #       description...
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # find the header "Parameters" and its underline
        if (
            line.strip() == "Parameters"
            and i + 1 < len(lines)
            and set(lines[i + 1].strip()) == {"-"}
        ):
            # Add Parameters header and underline
            out.append(line)
            out.append(lines[i + 1])
            i += 2

            # Parse parameter entries until we hit a new section or EOF
            parsed_params = []
            while i < len(lines):
                if not lines[i].strip():
                    # Empty line - could be end of parameters or just spacing
                    # Look ahead to see if next non-empty line is a new section
                    next_i = i + 1
                    while next_i < len(lines) and not lines[next_i].strip():
                        next_i += 1

                    if next_i < len(lines):
                        next_line = lines[next_i].strip()
                        # Check if it's a new section header (might have underline)
                        if (
                            next_line
                            and next_i + 1 < len(lines)
                            and set(lines[next_i + 1].strip()).issubset(
                                {"-", "="}
                            )
                            and lines[next_i + 1].strip()
                        ):
                            # It's a new section, stop parsing parameters
                            break

                    # Add empty line and continue
                    parsed_params.append(lines[i])
                    i += 1
                    continue

                # Check if this is a parameter entry (name : type format)
                if lines[i].strip() and ":" in lines[i]:
                    # Get the indentation of this line
                    current_indent = len(lines[i]) - len(lines[i].lstrip())
                    param_line = lines[i].strip()
                    param_type = (
                        param_line.split(":", 1)[1].strip()
                        if ":" in param_line
                        else ""
                    )
                    param_start = i
                    i += 1

                    # Consume the parameter's description (more indented lines)
                    while i < len(lines):
                        if not lines[i].strip():
                            # Empty line - keep it as part of this parameter
                            i += 1
                        elif (
                            len(lines[i]) - len(lines[i].lstrip())
                            > current_indent
                        ):
                            # More indented line - it's part of the description
                            i += 1
                        else:
                            # Same or less indented - end of this parameter
                            break

                    # Only remove the parameter if it's "lT" with "LineageTree" type
                    should_remove = (
                        "LineageTree" in param_type
                        or param_type == "LineageTree"
                    )

                    # If this is not the parameter to remove, keep it
                    if not should_remove:
                        parsed_params.extend(lines[param_start:i])
                else:
                    # Not a parameter entry - we've reached the end of parameters section
                    break

            # Add the remaining parameters
            out.extend(parsed_params)

            # Add the rest of the docstring
            out.extend(lines[i:])
            return "\n".join(out)
        else:
            out.append(line)
        i += 1

    return doc  # not numpy style


def methodize(func):
    """Wrap a free function so it behaves as a bound method.

    The returned wrapper drops the first parameter from the visible signature
    and hides the corresponding entry in the NumPy-style docstring, so that a
    free function ``func(self, *args)`` can be attached to a class as a method.

    Parameters
    ----------
    func : callable
        The free function to wrap. Its first parameter is treated as ``self``.

    Returns
    -------
    callable
        A wrapper suitable for assignment as a class attribute, e.g.
        ``Class.func = methodize(func)``.
    """

    @wraps(func)
    def _method(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    _method.__doc__ = _strip_first_param_from_doc(func.__doc__ or "")

    return _method


def attach_methods(cls, funcs):
    """Attach several methodised free functions to a class in bulk.

    Parameters
    ----------
    cls : type
        The class to which the methods are attached.
    funcs : module or dict of {str: callable}
        Either a module (its public, callable, non-underscored attributes are
        used) or a mapping from method name to free function.

    Raises
    ------
    TypeError
        If ``funcs`` is neither a module nor a dict of callables.
    """
    if hasattr(funcs, "__dict__"):
        items = {
            name: obj
            for name, obj in funcs.__dict__.items()
            if callable(obj) and not name.startswith("_")
        }
    elif isinstance(funcs, dict):
        items = funcs
    else:
        raise TypeError("funcs must be a module or dict of callables")

    for name, f in items.items():
        setattr(cls, name, methodize(f))


def _should(name: str, obj: object, cls_module: str | None) -> bool:
    """Decide whether ``obj`` should be automatically methodised.

    A callable is eligible when:

    - It is a plain :class:`types.FunctionType` (not a ``staticmethod``,
      ``classmethod``, or ``property``).
    - It was **not** defined inside the same module as the mixin class
      (i.e. it was imported from a ``_core`` module).
    - Its first parameter is positional (``POSITIONAL_ONLY`` or
      ``POSITIONAL_OR_KEYWORD``).

    Parameters
    ----------
    name : str
        Attribute name (unused; kept for caller convenience).
    obj : object
        The object assigned to the class attribute.
    cls_module : str or None
        ``__module__`` of the mixin class being constructed.

    Returns
    -------
    bool
        ``True`` if ``obj`` should be wrapped by :func:`methodize`.
    """
    if isinstance(obj, (staticmethod, classmethod, property)):
        return False
    if not isinstance(obj, types.FunctionType):
        return False
    if getattr(obj, "__module__", None) == cls_module:
        return False
    try:
        first = next(iter(inspect.signature(obj).parameters.values()))
    except StopIteration:
        return False
    return first.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


class AutoMethodizeMeta(type):
    """Metaclass that automatically wraps imported free functions as bound methods.

    When a mixin class is created with this metaclass, every function that
    satisfies :func:`_should` (or every name listed in the optional
    ``__methodize__`` class attribute) is replaced by a
    :func:`methodize`-wrapped version. The net effect is that free functions
    imported from the ``_core`` modules appear as ordinary instance methods
    on the :class:`~lineagetree.LineageTree` class.

    The ``lT : LineageTree`` first parameter is also stripped from the
    visible method signature and from the NumPy-style docstring.
    """

    def __new__(mcls, name, bases, ns, **kw):
        """Create the new mixin class and attach methodised functions.

        Parameters
        ----------
        name : str
            Class name.
        bases : tuple
            Base classes.
        ns : dict
            Class namespace.
        **kw
            Additional keyword arguments forwarded to :func:`type.__new__`.

        Returns
        -------
        type
            The newly created class with methodised functions attached.
        """
        cls = super().__new__(mcls, name, bases, ns, **kw)

        if "__methodize__" in ns:
            funcs = {n: getattr(cls, n) for n in ns["__methodize__"]}
        else:
            funcs = {
                n: getattr(cls, n)
                for n, v in ns.items()
                if _should(n, v, ns.get("__module__"))
            }

        if funcs:
            attach_methods(cls, funcs)
        return cls
