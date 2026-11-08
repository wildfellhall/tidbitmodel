"""Refresh shared implementations while preserving standalone settings and UI.

Run after changing tidbit/data.py or tidbit/training.py. The resulting
one_piece.py still contains everything needed in a single notebook cell.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def definitions(path):
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    return {
        node.name: "".join(lines[node.lineno - 1 : node.end_lineno])
        for node in ast.parse(text).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def sync():
    shared = definitions(ROOT / "tidbit/data.py")
    shared.pop("load_sources")
    training = definitions(ROOT / "tidbit/training.py")
    for name in ("read_prompts", "generate"):
        training.pop(name)
    shared.update(training)
    shared["prepare"] = (
        shared["prepare"]
        .replace("load_sources(args.corpus)", "load_sources(args.sources)")
        .replace(
            'write_json(staging / "metadata.json", metadata)',
            'metadata["preparation_signature"] = preparation_signature(args)\n'
            '            write_json(staging / "metadata.json", metadata)',
        )
    )
    target = ROOT / "one_piece.py"
    text = target.read_text()
    lines = text.splitlines(keepends=True)
    nodes = [
        n
        for n in ast.parse(text).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in shared
    ]
    for node in reversed(nodes):
        lines[node.lineno - 1 : node.end_lineno] = [
            shared.pop(node.name).rstrip() + "\n"
        ]
    text = "".join(lines)
    if shared:
        text = text.replace(
            "# Parsing and corpus preparation.",
            "\n\n".join(shared.values()) + "\n\n# Parsing and corpus preparation.",
            1,
        )
    corpus = (ROOT / "corpus.py").read_text()

    def source_node(source):
        return next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "sources" for t in n.targets)
        )

    original, replacement = source_node(text), source_node(corpus)
    lines = text.splitlines(keepends=True)
    embedded = corpus.splitlines(keepends=True)[
        replacement.lineno - 1 : replacement.end_lineno
    ]
    before = re.sub(r"\n{4,}", "\n\n\n", "".join(lines[: original.lineno - 1]))
    after = re.sub(r"\n{4,}", "\n\n\n", "".join(lines[original.end_lineno :]))
    embedded_text = "".join(line.rstrip() + "\n" for line in embedded)
    target.write_text(before + embedded_text + after)


if __name__ == "__main__":
    sync()
