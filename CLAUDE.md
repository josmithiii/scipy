# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is a clone of [SciPy](https://github.com/scipy/scipy) (`1.15.0.dev0`). JOS works on
a personal `jos` branch that tracks `main` via `/merge-upstream`. Upstream contribution
conventions (PR-per-feature, tests required, style gates below) apply, but on the `jos`
branch temporary scratch files are normal — see **Active `jos` work** below.

## Build & develop

SciPy is a compiled package (C/C++/Cython/Fortran/Pythran via **Meson / meson-python**). You
cannot just `import scipy` from a source checkout — it must be built first. Everything goes
through the `dev.py` orchestrator, which does an isolated in-tree build under `build/` and
installs to `build-install/`, then puts that on `sys.path` for you.

```bash
# One-time: install dev tooling (dev.py itself needs `doit`, `pydevtool`, `rich-click`)
pip install -r requirements/dev.txt        # or: pip install -e '.[dev,test]' (no build isolation)

python dev.py build                        # incremental build (re-run after editing compiled code)
python dev.py build -j8                     # parallel
python dev.py test -s signal                # test one submodule
python dev.py test -t scipy/signal/tests/test_filter_design.py    # one test file
python dev.py test -t scipy.optimize.tests.test_minimize_constrained   # dotted form OK too
python dev.py test -s stats -- -k some_pattern --tb=line   # args after `--` go to pytest
python dev.py test -j8 -s signal            # parallel (pytest-xdist)
python dev.py python script.py              # run a script against the built scipy
python dev.py ipython / python dev.py shell # REPL/shell with built scipy on PATH
python dev.py bench -t signal.Filtering     # asv benchmarks; `--compare main` to diff
```

Pure-Python edits are picked up on the next `dev.py test`/`python` without a manual rebuild;
edits to `.pyx`/`.c`/`.cpp`/`.pyf`/`meson.build` require `python dev.py build`.

## Lint & type checks (must pass before an upstream PR)

```bash
python dev.py lint                 # ruff + cython-lint (add --fix to auto-fix)
python dev.py mypy                 # mypy (pinned to 1.10.0 in [dev])
python tools/lint.py --diff-against=main    # what CI actually runs
```

Gates: ruff (`[tool.ruff]` in `pyproject.toml`), cython-lint, mypy (`mypy.ini`). CI also runs a
refguide/doctest check (`python dev.py refguide-check`, `python dev.py smoke-docs`).

## Architecture

`scipy/<subpackage>/` — each subpackage (`signal`, `optimize`, `stats`, `linalg`, `sparse`,
`special`, `interpolate`, `integrate`, `fft`, `ndimage`, `spatial`, …) is self-contained with:
- a public `__init__.py` whose module docstring is the API reference (numpydoc format);
- private implementation in `_underscore_prefixed.py` modules — the public names are
  re-exported from `__init__`. **New public API goes in a `_private` module + `__all__`.**
- `tests/` with pytest tests (`test_*.py`);
- a `meson.build` declaring every installed file and every compiled extension. **Adding a new
  `.py`, `.pyx`, or data file means editing that `meson.build`** or it won't be installed.

Compiled code: Cython (`.pyx`/`.pxd`), C/C++, Fortran (via `f2py`, `.pyf`), and Pythran
(`.py` with `#pythran export`). `scipy/_lib/` holds shared utilities (incl. vendored deps like
`boost_math` as a submodule — note `git status` shows it modified). `scipy/_build_utils/`
holds build helpers. Array-API support is tested with `array-api-strict` and alternate
backends (`python dev.py test -b torch -b numpy ...`).

## Active `jos` work: `invfreqz` for `scipy.signal`

The `jos` branch is developing **`invfreqz`** — an FFT-based equation-error / Steiglitz-McBride
method for designing an IIR filter `(B, A)` from a desired frequency response `H` (inverse of
`freqz`). Reference derivation:
https://ccrma.stanford.edu/~jos/filters/FFT_Based_Equation_Error_Method.html

All development lives in **temporary scratch files in `scipy/signal/`**, deliberately kept out
of a clean squash-merge. They are intended to be absorbed, in final form, into
`scipy/signal/_filter_design.py`:

- `invfreqz_jos.py` — the `invfreqz()` implementation (the thing being built).
- `test_invfreqz_jos.py` — a numbered test driver (Tests 1..34+). Run **all** tests or one:
  ```bash
  cd scipy/signal && python test_invfreqz_jos.py       # all tests
  cd scipy/signal && python test_invfreqz_jos.py 33     # just test 33
  ```
  Also runnable via `pytest` from the repo root (`cd /w/scipy && pytest --cache-clear`).
- Support modules, all `*_jos.py` in `scipy/signal/`: `spectrum_utilities_jos.py`,
  `filter_utilities_jos.py`, `filter_plot_utilities_jos.py`, `filter_test_utilities_jos.py`,
  `spectrum_plot_utilities_jos.py`, `array_utilities_jos.py`.

**Import gotcha:** these files use *bare* intra-directory imports
(`from spectrum_utilities_jos import ...`), so they only resolve when `scipy/signal/` is on
`sys.path` — i.e. run them from inside `scipy/signal/`, or let pytest's prepend-import insert
that dir. They import the *built* scipy (`from scipy.signal import freqz`), so `python dev.py
build` must have succeeded first. Other untracked scratch files here (`invfreqz.cpp`,
`invfreqz.lua`, `dgn.py`, `_filter_design_DIFFS*.py`) are experiments, not part of the build.

When the API stabilizes, the merge target is `_filter_design.py` (public re-export in
`scipy/signal/__init__.py` + `meson.build` — but for scratch `*_jos.py` files, do **not** add
them to `meson.build`; they are not meant to be installed).
