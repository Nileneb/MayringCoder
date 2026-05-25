"""#266: after the dedup, NOTHING in src/ may import the deleted src/agents
package. This gate fails until every consumer is swapped to mayring_pi_agent."""
import pathlib
import re

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
_PAT = re.compile(r"\b(from|import)\s+src\.agents\b|['\"]src\.agents")


def test_no_src_agents_references_in_src():
    offenders = []
    for py in _SRC.rglob("*.py"):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if _PAT.search(line):
                offenders.append(f"{py.relative_to(_SRC.parent)}:{i}")
    assert not offenders, "src.agents still referenced:\n" + "\n".join(offenders)
