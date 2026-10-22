"""Metadata helpers shared by the framework integrations' FilteredIndex use."""

from __future__ import annotations

import re

_SCALAR_TYPES = (str, int, float, bool, type(None))
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_KEYS = frozenset({"id", "position"})
_SQL_KEYWORDS = frozenset(
    """abort action add after all alter always analyze and as asc attach autoincrement before
    begin between by cascade case cast check collate column commit conflict constraint create
    cross current current_date current_time current_timestamp database default deferrable
    deferred delete desc detach distinct do drop each else end escape except exclude exclusive
    exists explain fail filter first following for foreign from full generated glob group groups
    having if ignore immediate in index indexed initially inner insert instead intersect into is
    isnull join key last left like limit match materialized natural no not nothing notnull null
    nulls of offset on or order others outer over partition plan pragma preceding primary query
    raise range recursive references regexp reindex release rename replace restrict returning
    right rollback row rows savepoint select set table temp temporary then ties to transaction
    trigger unbounded union unique update using vacuum values view virtual when where window with
    without""".split()  # noqa: SIM905 -- one readable block
)
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


def is_filterable_key(key: object) -> bool:
    """Whether `key` can be a SQLFilter metadata column."""
    return (
        isinstance(key, str)
        and _IDENTIFIER_RE.match(key) is not None
        and key.lower() not in _RESERVED_KEYS
        and key.lower() not in _SQL_KEYWORDS
    )


def is_sql_scalar(value: object) -> bool:
    """Whether SQLite stores `value` as itself (its integers are 64-bit)."""
    if isinstance(value, int) and not isinstance(value, bool):
        return _INT64_MIN <= value <= _INT64_MAX
    return isinstance(value, _SCALAR_TYPES)


def filterable_metadata(meta: dict | None) -> dict:
    """The part of a framework metadata dict the SQL filter can index.

    Keeps identifier-safe keys whose value is a scalar (str / int / float /
    bool / None) or a list of non-null scalars (multi-valued). Anything else
    -- nested dicts, objects, keys like ``"file-name"`` or ``"position"``,
    integers beyond 64 bits -- stays on the returned Document / node but
    can't be used in ``filter=``.
    """
    out: dict = {}
    for key, value in (meta or {}).items():
        if not is_filterable_key(key):
            continue
        if is_sql_scalar(value):
            out[key] = value
        elif isinstance(value, (list, tuple, set)) and all(
            is_sql_scalar(v) and v is not None for v in value
        ):
            out[key] = list(value)
    return out


def replacement_patch(old: dict | None, new: dict | None) -> dict:
    """A SQLFilter.update() patch that *replaces* `old` metadata with `new`:
    keys `new` drops are cleared (None for scalars, [] for lists)."""
    patch: dict = {
        key: [] if isinstance(value, list) else None
        for key, value in filterable_metadata(old).items()
    }
    patch.update(filterable_metadata(new))
    return patch
