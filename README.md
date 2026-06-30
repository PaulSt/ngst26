# NGSTrefftz Book

## Quick Start

Run:

```sh
make install
make start
```

`make install` creates a local `.venv` folder and installs the Python requirements there.
`make start` converts the MyST markdown chapters in `chapters/` to generated `.ipynb` notebooks in `notebooks/` and opens JupyterLab at that folder.
If `.venv` is missing, `make start` creates it automatically.

If you are on Windows without `make`, use the same helper directly:

```sh
python scripts/book_tools.py install
python scripts/book_tools.py start
```

