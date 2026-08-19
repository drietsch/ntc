"""Extract NTC RawToolSchema JSON from Pimcore agent-bundle MCP tool classes.

Parses the `#[McpTool(name:, description:)]` class attribute and the
`execute()` signature's per-parameter `#[Schema(...)]` attributes
(php-mcp/server attribute style). V1 keeps scalar/enum parameters only;
array/object parameters are dropped (noted), and tools whose REQUIRED
parameters can't be represented are excluded.

Run: uv run python -m tools.extract_pimcore_tools --src /tmp/pimcore-tools \
        --out ../examples/pimcore-tools.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TYPE_MAP = {"string": "string", "integer": "integer", "int": "integer",
            "number": "number", "float": "number", "boolean": "boolean", "bool": "boolean"}

# POC brevity cap: keep leading sentences up to this many chars (the canonical
# renderer caps at 200 anyway; shorter descriptions keep schema sequences
# within the model's Ls window).
MAX_DESC = 140
MAX_PARAM_DESC = 100


def cap_desc(text: str, limit: int = MAX_DESC) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    out = ""
    for m in re.finditer(r"[^.!?]*[.!?]", text):
        if out and len(out) + len(m.group(0)) > limit:
            break
        out += m.group(0)
        if len(out) >= limit:
            break
    out = out.strip() or text[:limit]
    return out[:limit]

# V1 model limit (backbone arch): at most 8 declared args per tool.
MAX_ARGS = 8


def parse_php_array(expr: str | None) -> dict | None:
    """Parse a PHP array literal like `['type' => 'integer']` (one nesting
    level of `properties`/`required` included) into a JSON-Schema-ish dict."""
    if not expr:
        return None
    expr = expr.strip()
    if not expr.startswith("["):
        return None
    out: dict = {}
    # Top-level `key => value` pairs.
    depth, key, buf, pairs = 0, None, [], []
    for ch in expr[1:-1]:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            pairs.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        pairs.append("".join(buf))
    for pair in pairs:
        if "=>" not in pair:
            continue
        k, v = pair.split("=>", 1)
        key = parse_php_string(k)
        v = v.strip()
        if v.startswith("["):
            nested = parse_php_array(v)
            if nested is not None:
                out[key] = nested
                continue
            out[key] = [parse_php_string(x) for x in re.findall(r"'[^']*'", v)]
        else:
            out[key] = parse_php_string(v)
    return out or None


def parse_php_string(expr: str) -> str:
    """Concatenated PHP string literal ('a' . 'b' . "c") → python str."""
    out: list[str] = []
    for m in re.finditer(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"", expr):
        s = m.group(1) if m.group(1) is not None else m.group(2)
        out.append(s.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\"))
    return "".join(out)


def balanced(text: str, start: int) -> str:
    """Return the contents of the (...) starting at text[start] == '('."""
    depth, i = 0, start
    while i < len(text):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
        elif c in "'\"":
            q = c
            i += 1
            while i < len(text) and text[i] != q:
                i += 2 if text[i] == "\\" else 1
        i += 1
    raise ValueError("unbalanced parens")


def named_arg(body: str, name: str) -> str | None:
    m = re.search(rf"\b{name}\s*:", body)
    if not m:
        return None
    rest = body[m.end():]
    # capture until a top-level comma
    depth = 0
    for i, c in enumerate(rest):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c in "'\"":
            q = c
            j = i + 1
            while j < len(rest) and rest[j] != q:
                j += 2 if rest[j] == "\\" else 1
            continue
        elif c == "," and depth == 0:
            return rest[:i].strip()
    return rest.strip()


def extract(path: Path) -> tuple[dict | None, list[str]]:
    src = path.read_text()
    notes: list[str] = []
    m = re.search(r"#\[McpTool\(", src)
    if not m:
        return None, [f"{path.name}: no McpTool attribute"]
    tool_attr = balanced(src, m.end() - 1)
    name_expr = named_arg(tool_attr, "name")
    desc_expr = named_arg(tool_attr, "description")
    if not name_expr:
        return None, [f"{path.name}: no tool name"]
    tool_name = parse_php_string(name_expr)
    description = cap_desc(parse_php_string(desc_expr)) if desc_expr else ""

    # execute(...) signature after the McpTool attribute.
    sig_m = re.search(r"function\s+execute\s*\(", src[m.end():])
    if not sig_m:
        return None, [f"{path.name}: no execute()"]
    sig = balanced(src[m.end():], sig_m.end() - 1)

    params: dict[str, dict] = {}
    # Split signature at top-level commas.
    parts, depth, cur = [], 0, []
    i = 0
    while i < len(sig):
        c = sig[i]
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c in "'\"":
            q = c
            cur.append(c)
            i += 1
            while i < len(sig) and sig[i] != q:
                cur.append(sig[i])
                i += 2 if sig[i] == "\\" else 1
            cur.append(sig[i] if i < len(sig) else q)
            i += 1
            continue
        if c == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        parts.append("".join(cur))

    for part in parts:
        pm = re.search(r"(\??)([A-Za-z_|\\\[\]]+)\s+(?:&\s*)?(?:\.\.\.)?\$(\w+)\s*(=)?", part)
        if not pm:
            continue
        nullable, php_type, pname, has_default = pm.groups()
        schema_m = re.search(r"#\[Schema\(", part)
        s_type, s_desc, s_enum = None, "", None
        if schema_m:
            body = balanced(part, schema_m.end() - 1)
            t = named_arg(body, "type")
            s_type = parse_php_string(t) if t else None
            d = named_arg(body, "description")
            s_desc = parse_php_string(d) if d else ""
            e = named_arg(body, "enum")
            if e:
                s_enum = [parse_php_string(x) for x in re.findall(r"'[^']*'|\"[^\"]*\"", e)]
        eff_type = s_type or php_type.lower().lstrip("?")
        required = not (nullable or has_default)

        # Composite parameters (spec §19 LIST<T>/OBJECT<T>): pass `items` /
        # `properties` through so the Rust schema compiler can decide between
        # LIST<scalar>, flattened object, and OPAQUE (agent-only).
        if eff_type in ("array", "object"):
            p = {"type": "array" if eff_type == "array" else "object"}
            if s_desc:
                p["description"] = cap_desc(s_desc, MAX_PARAM_DESC)
            items = parse_php_array(named_arg(body, "items")) if schema_m else None
            props = parse_php_array(named_arg(body, "properties")) if schema_m else None
            if items is not None:
                p["items"] = items
            if props is not None:
                p["properties"] = props
            if required:
                p["required"] = True
            params[pname] = p
            kind = "LIST" if items and isinstance(items.get("type"), str) and items["type"] != "object" else "OPAQUE"
            notes.append(f"{tool_name}: `{pname}` -> {eff_type} ({kind})")
            continue

        if eff_type not in TYPE_MAP:
            notes.append(f"{tool_name}: drop param `{pname}` (type {eff_type})")
            if required:
                return None, [f"{tool_name}: required param `{pname}` has unsupported type {eff_type}"]
            continue
        p: dict = {"type": TYPE_MAP[eff_type]}
        if s_desc:
            p["description"] = cap_desc(s_desc, MAX_PARAM_DESC)
        if s_enum:
            if len(s_enum) > 4:
                notes.append(f"{tool_name}: enum `{pname}` truncated {len(s_enum)}→4")
                s_enum = s_enum[:4]
            p["enum"] = s_enum
        if required:
            p["required"] = True
        params[pname] = p

    if len(params) > MAX_ARGS:
        required_first = sorted(params.items(), key=lambda kv: not kv[1].get("required"))
        notes.append(f"{tool_name}: {len(params)} params trimmed to {MAX_ARGS}")
        params = dict(required_first[:MAX_ARGS])

    return {"name": tool_name, "description": description, "parameters": params}, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("/tmp/pimcore-tools"))
    parser.add_argument("--out", type=Path, default=Path("../examples/pimcore-tools.json"))
    args = parser.parse_args()

    tools, skipped = [], []
    for path in sorted(args.src.rglob("*Tool.php")):
        schema, notes = extract(path)
        for n in notes:
            print("  note:", n)
        if schema is None:
            skipped.append(path.name)
            continue
        tools.append(schema)
    print(f"extracted {len(tools)} tools, skipped {len(skipped)}: {skipped}")
    args.out.write_text(json.dumps(tools, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
