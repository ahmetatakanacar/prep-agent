"""Turns a pydantic model's JSON schema into one OpenAI's strict structured outputs accepts.

Strict mode takes a subset of JSON Schema. Pydantic's own schema falls outside it in a few ways:
discriminated unions come out as `oneOf` plus `discriminator`, defaults appear as `default`, and
optional fields are not listed in `required`. The conversion is generated from the model, never
hand-written, so it cannot drift from it silently (tests check this).

Strict mode forces the model to write every property. A field that was optional therefore becomes
nullable, and a null the model sends for it means "not provided". `drop_added_nulls` removes those
nulls from the answer so the original model's own defaults apply and the answer can be validated
by the original model.
"""

import copy
from typing import Any

from pydantic import BaseModel

_DROPPED_KEYWORDS = {"default", "discriminator"}
# Values of these keywords are data, not schemas, and must not be walked or rewritten.
_DATA_KEYWORDS = {"enum", "const", "examples"}
# Values of these keywords map names to schemas, so their keys are never keywords themselves.
_NAME_MAPS = {"properties", "$defs"}
_NULL = {"type": "null"}


class UnsupportedSchemaError(ValueError):
    """The model uses a schema shape strict mode cannot express."""


def to_strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a new JSON schema for `model` that fits OpenAI's strict mode."""
    schema = model.model_json_schema()
    return _convert(schema, schema.get("$defs", {}), model.__name__)


def drop_added_nulls(data: Any, model: type[BaseModel]) -> Any:
    """Remove nulls that `to_strict_schema` made acceptable for fields that are not nullable.

    Only a null for a field that is optional in `model` and does not allow null itself is
    removed, so that the field's default applies. Anything that cannot be attributed to exactly
    one object schema (an unknown `op`, say) is left alone for validation to report. `data` is
    not modified.
    """
    schema = model.model_json_schema()
    return _prune(data, schema, schema.get("$defs", {}))


def _convert(node: Any, defs: dict[str, Any], where: str) -> Any:
    if isinstance(node, list):
        return [_convert(item, defs, where) for item in node]
    if not isinstance(node, dict):
        return node
    node = _unwrap_all_of(node)
    if "allOf" in node:
        raise UnsupportedSchemaError(f"{where}: allOf with several members is not supported")

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROPPED_KEYWORDS:
            continue
        if key in _NAME_MAPS:
            out[key] = {
                name: _convert(sub, defs, name if key == "properties" else where)
                for name, sub in value.items()
            }
        elif key in _DATA_KEYWORDS:
            out[key] = copy.deepcopy(value)
        elif key == "oneOf":
            if "anyOf" in node:
                raise UnsupportedSchemaError(f"{where}: oneOf next to anyOf is not supported")
            out["anyOf"] = _convert(value, defs, where)
        else:
            out[key] = _convert(value, defs, where)

    if "properties" in out:
        if node.get("additionalProperties") not in (None, False):
            raise UnsupportedSchemaError(f"{where}: extra properties are not supported")
        required = set(node.get("required", []))
        out["properties"] = {
            name: sub if name in required else _make_nullable(sub, defs)
            for name, sub in out["properties"].items()
        }
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    elif out.get("type") == "object":
        raise UnsupportedSchemaError(f"{where}: an object without declared properties")
    return out


def _make_nullable(schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    if _allows_null(schema, defs):
        return schema
    if "anyOf" in schema:
        return {**schema, "anyOf": [*schema["anyOf"], _NULL]}
    # Annotations go on the outer schema so that a bare $ref stays a bare $ref.
    outer = {key: schema[key] for key in ("title", "description") if key in schema}
    inner = {key: value for key, value in schema.items() if key not in outer}
    return {**outer, "anyOf": [inner, _NULL]}


def _allows_null(schema: dict[str, Any], defs: dict[str, Any]) -> bool:
    schema = _deref(schema, defs)
    kind = schema.get("type")
    if kind == "null" or (isinstance(kind, list) and "null" in kind):
        return True
    if "const" in schema and schema["const"] is None:
        return True
    if None in schema.get("enum", []):
        return True
    return any(
        _allows_null(member, defs) for member in schema.get("anyOf", []) + schema.get("oneOf", [])
    )


def _deref(schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    schema = _unwrap_all_of(schema)
    ref = schema.get("$ref")
    if ref is None:
        return schema
    name = ref.removeprefix("#/$defs/")
    if not ref.startswith("#/$defs/") or name not in defs:
        raise UnsupportedSchemaError(f"unresolvable reference {ref!r}")
    return _unwrap_all_of(defs[name])


def _unwrap_all_of(schema: dict[str, Any]) -> dict[str, Any]:
    members = schema.get("allOf")
    if isinstance(members, list) and len(members) == 1 and isinstance(members[0], dict):
        rest = {key: value for key, value in schema.items() if key != "allOf"}
        return {**members[0], **rest}
    return schema


def _members(schema: dict[str, Any], defs: dict[str, Any]) -> list[dict[str, Any]]:
    schema = _deref(schema, defs)
    union = schema.get("anyOf", []) + schema.get("oneOf", [])
    if not union:
        return [schema]
    return [member for option in union for member in _members(option, defs)]


def _prune(data: Any, schema: dict[str, Any], defs: dict[str, Any]) -> Any:
    members = _members(schema, defs)
    if isinstance(data, dict):
        chosen = _choose_object(data, [m for m in members if "properties" in m])
        if chosen is None:
            return data
        required = set(chosen.get("required", []))
        out = {}
        for key, value in data.items():
            sub = chosen["properties"].get(key)
            if sub is None:
                out[key] = value
            elif value is None and key not in required and not _allows_null(sub, defs):
                continue
            else:
                out[key] = _prune(value, sub, defs)
        return out
    if isinstance(data, list):
        arrays = [m for m in members if m.get("type") == "array" and "items" in m]
        if len(arrays) != 1:
            return data
        return [_prune(item, arrays[0]["items"], defs) for item in data]
    return data


def _choose_object(data: dict[str, Any], objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(objects) == 1:
        return objects[0]
    # Union members are told apart by their literal fields (the `op` of a plan step).
    matching = [
        candidate
        for candidate in objects
        if all(
            key in data and data[key] == sub["const"]
            for key, sub in candidate["properties"].items()
            if "const" in sub
        )
    ]
    return matching[0] if len(matching) == 1 else None
