# Contributing to OpenEBM

Thanks for your interest in improving OpenEBM. This project is research-grade
code, so our goals are: **clean, modular, well-documented** implementations
that other researchers can pick up and extend.

## Scope

OpenEBM is split into two zones:

| Zone | Policy |
|---|---|
| `openebm/elm/` | Active development happens here. Contributions are welcome. |
| `nanochat/` | Vendored upstream dependency — **do not modify**. Sync with upstream instead. |
| Other `openebm/*` sub-packages | Managed by their maintainers. Open an issue before sending patches. |

## Development workflow

1. Fork the repository and create a topic branch.
2. Install the dev dependencies:
   ```bash
   uv sync --extra gpu --group dev
   ```
3. Make your changes inside `openebm/elm/`.
4. Before committing:
   ```bash
   ruff check openebm/elm
   mypy openebm/elm
   pytest -m "not slow"
   ```
5. Open a pull request describing the **problem**, the **approach**, and
   the **empirical results** (if the change is algorithmic).

## Coding style

- Follow the existing structure. Prefer `typing.Optional`, `typing.List`,
  etc. over PEP 604 syntax to stay consistent.
- Use **Sphinx reST** docstrings:
  ```python
  def foo(x: int) -> int:
      """Return twice the input.

      :param x: integer operand
      :type x: int
      :return: ``2 * x``
      :rtype: int
      """
      return 2 * x
  ```
- Don't modify the commented-out exploratory blocks unless you are
  implementing the feature they describe.
- Don't add large binary artefacts to the repository.

## Issues and feature requests

Open a GitHub issue with:

- a short reproducible example (when reporting a bug), or
- a brief motivation + sketch of the API (when proposing a feature).

## License

By contributing, you agree that your contributions will be licensed under
the project's [MIT License](LICENSE).
