0. Ensure you have python installed, and at least Python 3.10: https://www.python.org/downloads/

1. Install Poetry: https://python-poetry.org/docs/#installation

2. Install `my_first_ltr` as local package using poetry:
```sh
poetry install
```
3.  Install pre-commit hooks
- nbstripout prevent pushing the output of jupyter notebooks.
- black for formatting

```
pre-commit install
```
