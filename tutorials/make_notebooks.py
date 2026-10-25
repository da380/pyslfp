"""Generates the tutorial notebooks from the tutorial scripts.

The scripts in `tutorials/scripts` are the source of the tutorials. They
are written in cells, the `# %%` convention that editors such as VS Code
and Spyder run cell by cell, and the test suite runs them headless. Each
is converted here into a notebook in `tutorials/notebooks`, for working
in Jupyter or trying the library online in Google Colab. A notebook gains
one cell, before its first code cell, that installs the library when it
runs on Colab and does nothing elsewhere.

The conversion is mechanical and deterministic. A `# %% [markdown]` cell
becomes a markdown cell with its comment markers removed, any other
`# %%` cell a code cell, and the cell ids are numbered as nbstripout
numbers them, so an unchanged script regenerates its notebook byte for
byte. A notebook whose script has gone is deleted. Run it by hand,

    poetry run python tutorials/make_notebooks.py

or let the pre-commit hook run it. It exits with status 1 when it has
changed anything, so that a commit with stale notebooks stops and the
regenerated notebooks can be added.
"""

import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
NOTEBOOKS = HERE / "notebooks"

MARKER = "# %%"

INSTALL = """\
# Generated from tutorials/scripts/{name}.
# Edit the script, not this notebook. On Google Colab the library is
# installed first; elsewhere this cell does nothing.
import subprocess
import sys

if "google.colab" in sys.modules:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "pyslfp"], check=True
    )"""

METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def _cells(script: Path) -> list[tuple[str, str]]:
    """The script's cells as (kind, source) pairs, kind 'markdown' or
    'code', with empty cells dropped."""
    blocks: list[tuple[str, list[str]]] = []
    for number, line in enumerate(script.read_text().splitlines(), start=1):
        if line.startswith(MARKER):
            tag = line[len(MARKER) :].strip()
            if tag not in ("", "[markdown]"):
                raise ValueError(f"{script.name}:{number}: unknown cell tag {tag!r}")
            blocks.append(("markdown" if tag else "code", []))
        elif blocks:
            blocks[-1][1].append(line)
        elif line.strip():
            raise ValueError(f"{script.name}:{number}: text before the first cell")
    cells = []
    for kind, lines in blocks:
        if kind == "markdown":
            text = []
            for line in lines:
                if line.strip() and line != "#" and not line.startswith("# "):
                    raise ValueError(
                        f"{script.name}: markdown line not commented: {line!r}"
                    )
                text.append(line[2:] if line.startswith("# ") else "")
            lines = text
        source = "\n".join(lines).strip("\n")
        if source.strip():
            cells.append((kind, source))
    return cells


def notebook(script: Path) -> str:
    """The text of the notebook generated from a script."""
    cells = _cells(script)
    first_code = next((i for i, (kind, _) in enumerate(cells) if kind == "code"), 0)
    cells.insert(first_code, ("code", INSTALL.format(name=script.name)))
    nb = nbformat.v4.new_notebook(metadata=METADATA)
    for i, (kind, source) in enumerate(cells):
        if kind == "markdown":
            cell = nbformat.v4.new_markdown_cell(source)
        else:
            cell = nbformat.v4.new_code_cell(source)
        cell["id"] = str(i)
        nb.cells.append(cell)
    nbformat.validate(nb)
    text = nbformat.writes(nb)
    return text if text.endswith("\n") else text + "\n"


def main() -> int:
    NOTEBOOKS.mkdir(exist_ok=True)
    changed = []
    wanted = set()
    for script in sorted(SCRIPTS.glob("*.py")):
        target = NOTEBOOKS / (script.stem + ".ipynb")
        wanted.add(target)
        text = notebook(script)
        if not target.exists() or target.read_text() != text:
            target.write_text(text)
            changed.append(f"wrote {target.relative_to(HERE.parent)}")
    for orphan in sorted(set(NOTEBOOKS.glob("*.ipynb")) - wanted):
        orphan.unlink()
        changed.append(f"deleted {orphan.relative_to(HERE.parent)}")
    for line in changed:
        print(line)
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
