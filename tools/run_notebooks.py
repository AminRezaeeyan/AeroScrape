"""Execute the analysis notebooks in place, so they are committed with their outputs.

    python tools/run_notebooks.py                  # every notebook in notebooks/
    python tools/run_notebooks.py notebooks/03_association_rules.ipynb

Each runs from notebooks/ (for nbutils) and fails loudly if a cell raises. Needs `make data`.
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def run(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOKS)}}
    ).execute()
    nbformat.write(notebook, path)
    print(f"executed {path.name}")


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or sorted(NOTEBOOKS.glob("*.ipynb"))
    for path in paths:
        run(path.resolve())
