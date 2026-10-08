"""Haystack filter dicts -> simlar MetadataFilter conditions.

Haystack's filter semantics (``haystack.utils.filters.document_matches_filter``)
are Python's: a missing field is ``None``, ``!=`` / ``not in`` match it, and
strings are compared as ISO dates by the ordering operators. SQL has
three-valued NULL logic and no date parsing, so each condition is emitted
*two-valued* (never NULL) and only where SQL provably agrees with Haystack is
it marked exact. Everything else compiles to a superset of the matching
documents, which the store then narrows with ``document_matches_filter`` --
so results always follow Haystack's semantics, and the common case is a
single cached SQL query.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from haystack import Document
from haystack.errors import FilterError
from haystack.utils.filters import COMPARISON_OPERATORS, LOGICAL_OPERATORS

from simlar.indexes.filtered_index import MetadataFilter
from simlar.integrations._filtering import is_filterable_key, is_sql_scalar

_ORDERING = frozenset({">", ">=", "<", "<="})
_SQL_OPS = {"==": "=", "!=": "!=", ">": ">", ">=": ">=", "<": "<", "<=": "<="}
_DOCUMENT_FIELDS = frozenset(f.name for f in fields(Document))
# Store-internal SQL columns start with this; user keys that do are kept off SQL.
_RESERVED_PREFIX = "_simlar"

# (filter, exact): filter None means "no restriction" (TRUE).
_Compiled = tuple[MetadataFilter | None, bool]


def _false() -> MetadataFilter:
    return MetadataFilter().where_null("id")  # ids are never NULL


def _and(parts: list[MetadataFilter]) -> MetadataFilter:
    return parts[0].and_(*parts[1:])


def _or(parts: list[MetadataFilter]) -> MetadataFilter:
    return parts[0].or_(*parts[1:])


@dataclass
class MetaFields:
    """What the store knows about each metadata key, so the compiler can tell
    whether SQL holds a key's values faithfully.

    Only grows: deleting documents never removes a kind, which can only make
    a condition inexact (rechecked), never wrong.
    """

    # key -> kinds seen: "num" (int/float/bool), "str", "other" (not in SQL as itself).
    kinds: dict[str, set[str]] = field(default_factory=dict)
    # lowercase -> key actually used as a SQL column (SQLite columns are case-insensitive).
    columns: dict[str, str] = field(default_factory=dict)

    def sql_metadata(self, meta: dict[str, Any]) -> dict[str, Any]:
        """Record `meta`'s keys and return the part written to the SQL filter."""
        out: dict[str, Any] = {}
        for key, value in meta.items():
            kinds = self.kinds.setdefault(key, set())
            if not self._claim(key) or not is_sql_scalar(value):
                kinds.add("other")
                continue
            out[key] = value
            if isinstance(value, str):
                kinds.add("str")
            elif value is not None:
                kinds.add("num")
        return out

    def _claim(self, key: object) -> bool:
        """Whether `key` gets (or already has) its own SQL column."""
        if not isinstance(key, str) or not is_filterable_key(key):
            return False
        if key.lower().startswith(_RESERVED_PREFIX):
            return False
        return self.columns.setdefault(key.lower(), key) == key

    def column(self, key: str) -> str | None:
        """The SQL column holding `key` if every document's value is in it as itself."""
        kinds = self.kinds.get(key)
        if kinds is None or "other" in kinds or self.columns.get(key.lower()) != key:
            return None
        return key


def compile_filters(filters: dict[str, Any], meta_fields: MetaFields) -> _Compiled:
    """Translate a Haystack filter dict into ``(MetadataFilter, exact)``.

    Raises FilterError for malformed filters, as ``document_matches_filter`` does.
    """
    if not isinstance(filters, dict):
        raise FilterError(f"Filters must be a dict, got {type(filters).__name__}.")
    if "field" in filters:
        return _comparison(filters, meta_fields)
    return _logic(filters, meta_fields)


def _logic(cond: dict[str, Any], meta_fields: MetaFields) -> _Compiled:
    if "operator" not in cond:
        raise FilterError(f"'operator' key missing in {cond}")
    if "conditions" not in cond:
        raise FilterError(f"'conditions' key missing in {cond}")
    op = cond["operator"]
    if op not in LOGICAL_OPERATORS:
        raise FilterError(
            f"Unknown logical operator '{op}'. Valid operators are: {sorted(LOGICAL_OPERATORS)}"
        )
    conditions = cond["conditions"]
    if not isinstance(conditions, list):
        raise FilterError(f"'conditions' must be a list in {cond}")
    parts = [compile_filters(c, meta_fields) for c in conditions]
    exact = all(e for _, e in parts)
    nodes = [n for n, _ in parts if n is not None]

    if op == "OR":
        if len(nodes) < len(parts) or not parts:
            # A TRUE branch makes the OR TRUE; an empty OR is FALSE.
            return (None, exact) if parts else (_false(), True)
        return _or(nodes), exact

    conjunction = _and(nodes) if nodes else None
    if op == "AND":
        return conjunction, exact
    # NOT is "not all of conditions". Negating a superset gives a subset, so
    # an inexact operand widens the whole NOT to TRUE (rechecked).
    if not exact:
        return None, False
    return (_false() if conjunction is None else ~conjunction), True


def _comparison(cond: dict[str, Any], meta_fields: MetaFields) -> _Compiled:
    if "operator" not in cond:
        raise FilterError(f"'operator' key missing in {cond}")
    if "value" not in cond:
        raise FilterError(f"'value' key missing in {cond}")
    name, op, value = cond["field"], cond["operator"], cond["value"]
    if op not in COMPARISON_OPERATORS:
        raise FilterError(
            f"Unknown comparison operator '{op}'. Valid operators are: {sorted(COMPARISON_OPERATORS)}"
        )
    if op in ("in", "not in") and not isinstance(value, list):
        raise FilterError(
            f"Filter value must be a `list` when using operator 'in' or 'not in', "
            f"received type '{type(value)}'"
        )
    if op in _ORDERING and isinstance(value, list):
        raise FilterError(
            f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        )
    if not isinstance(name, str):
        raise FilterError(f"'field' must be a string in {cond}")

    if name == "id":
        column, kinds = "id", {"str"}
    else:
        key = _meta_key(name)
        if key is None:  # content, embedding, nested meta.a.b, ...
            return None, False
        if key not in meta_fields.kinds:
            # No document has ever had this key: every value is None.
            matches = COMPARISON_OPERATORS[op](
                value=None, filter_value=value, strict_datetime_comparison=False
            )
            return (None if matches else _false()), True
        trusted = meta_fields.column(key)
        if trusted is None:
            return None, False
        column, kinds = trusted, meta_fields.kinds[key]

    if op in ("==", "!="):
        node, exact = _equals(column, value)
        if op == "==":
            return node, exact
        return (~node, True) if exact and node is not None else (None, False)
    if op in ("in", "not in"):
        node, exact = _one_of(column, value)
        if op == "in":
            return node, exact
        return (~node, True) if exact and node is not None else (None, False)
    # Ordering: Haystack is False for None on either side.
    if value is None:
        return _false(), True
    if _is_number(value) and kinds <= {"num"}:
        return _not_null(column).where(column, _SQL_OPS[op], value), True
    return _not_null(column), False


def _meta_key(name: str) -> str | None:
    """The metadata key a filter field refers to, or None if it isn't a flat
    metadata key. Fields without the ``meta.`` prefix that aren't Document
    attributes are metadata keys too (legacy filters), as in Haystack."""
    if name.startswith("meta."):
        key = name[5:]
        return None if "." in key else key
    if "." in name or name in _DOCUMENT_FIELDS:
        return None
    return name


def _not_null(column: str) -> MetadataFilter:
    return ~MetadataFilter().where_null(column)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and is_sql_scalar(value) and value == value  # not NaN


def _exact_value(value: object) -> bool:
    """Whether SQL `=` on `value` agrees with Haystack's `==`, which also
    treats two spellings of the same ISO date as equal."""
    if isinstance(value, str):
        return not _looks_like_iso_date(value)
    return _is_number(value)


def _equals(column: str, value: object) -> _Compiled:
    if value is None:
        return MetadataFilter().where_null(column), True
    if not _exact_value(value):
        return _not_null(column), False
    return _not_null(column).where(column, "=", value), True


def _one_of(column: str, values: list) -> _Compiled:
    scalars = [v for v in values if v is not None]
    if not all(_exact_value(v) for v in scalars):
        return None, False
    parts: list[MetadataFilter] = []
    if scalars:
        parts.append(_not_null(column).where(column, "IN", scalars))
    if len(scalars) < len(values):
        parts.append(MetadataFilter().where_null(column))
    return (_or(parts) if parts else _false()), True


def _looks_like_iso_date(value: str) -> bool:
    # Mirrors haystack.utils.filters._looks_like_iso_date (a private helper).
    return (
        len(value) >= 10
        and value[:4].isdigit()
        and value[4] == "-"
        and value[5:7].isdigit()
        and value[7] == "-"
        and value[8:10].isdigit()
        and (len(value) == 10 or value[10] in {"T", "t", " "})
    )
