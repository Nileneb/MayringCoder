import re


_IMPORT_RE = re.compile(
    r"^(?:from\s+([\w.]+)\s+import\s+|import\s+)(.+?)$", re.MULTILINE
)
_METHOD_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)
_CLASS_RE = re.compile(r"^\s*class\s+(\w+)", re.MULTILINE)


def extract_python_signatures(content: str) -> dict:
    """Extract function names, class names, and imports from Python source.

    Returns:
        {
            "functions": ["handle_request", "create_user", ...],
            "classes":   ["UserController", "Service", ...],
            "imports":   ["django.http", "rest_framework", ...],
        }
    """
    imports: list[str] = []
    for m in _IMPORT_RE.finditer(content):
        if m.group(1):
            imports.append(m.group(1).strip())
        else:
            for part in m.group(2).split(","):
                imports.append(part.strip().split(" as ")[0].strip())

    return {
        "functions": _METHOD_RE.findall(content),
        "classes": _CLASS_RE.findall(content),
        "imports": [i for i in imports if i],
    }
