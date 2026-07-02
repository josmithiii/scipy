# TODO: Completing the `invfreqz` PR for scipy.signal

Roadmap from the current scratch state (`scipy/signal/*_jos.py` on the `jos` branch) to an
upstreamable SciPy pull request. Ordered roughly by dependency: decisions first, then code,
tests, docs, and finally PR mechanics.

Reference derivation: https://ccrma.stanford.edu/~jos/filters/FFT_Based_Equation_Error_Method.html

---

## 0. Process prerequisites (SciPy requires these for new public API)

- [ ] **Open a feature proposal** before (or alongside) the PR: a GitHub issue on
      scipy/scipy labeled `enhancement` describing the API, and/or a short post to the
      scipy-dev mailing list / Scientific Python Discourse. New public functions are
      expected to get API feedback *before* review of the implementation. Search first
      for existing "invfreqz" / "frequency-domain IIR fit" issues and link them --
      MATLAB-parity requests for `invfreqz`/`invfreqs` have come up before.
- [ ] **One clean PR, one feature.** The final PR should contain only: the new code in
      `_filter_design.py`, tests, and doc plumbing. None of the scratch files
      (`*_jos.py`, `invfreqz.cpp`, `invfreqz.lua`, `dgn.py`, `_filter_design_DIFFS*.py`,
      this TODO, `CLAUDE.md` deltas) may appear in it. Build the PR branch fresh off
      upstream `main` and squash-apply the final form.

## 1. API decisions to finalize (blockers for everything else)

- [ ] **Trim the signature to what is implemented.** Ship v1 with only the working paths:
      equation-error (direct) and Steiglitz-McBride (iterative). Drop from the public
      signature until implemented: `method='prony'/'pade_prony'`, `method_iter='gauss_newton'`,
      `weight` (output-error weighting). SciPy convention is to *not* expose
      `NotImplementedError` placeholders in a new function's API -- options can be added
      later without deprecation, removal cannot. This likely collapses `method`/`method_iter`
      into a single `method` parameter, or none at all (`n_iter=0` → equation error,
      `n_iter>0` → Steiglitz-McBride).
- [ ] **Parameter naming/consistency with the `signal` namespace** (final names need
      reviewer buy-in):
      - `H, n_zeros, n_poles` vs. MATLAB's `(h, w, nb, na)` -- keep descriptive names, but
        consider accepting the frequency grid as `worN`/`w` the way `freqz` returns it.
      - `omega` → probably `w` (matches `freqz`/`freqs`), plus an optional `fs` parameter
        (SciPy signal functions accept `fs` and default to radians/sample).
      - Decide whether `U` (input/weighting spectrum) is clear enough or should be
        `weights`-like; document its equation-error-weighting role precisely.
- [ ] **Grid contract.** Current implementation requires a uniform grid from dc to π
      *inclusive*, length ideally 2^k + 1. Decide: hard-require (ValueError otherwise,
      fail fast) or resample internally. Document the FFT-based reason.
      MATLAB `invfreqz` accepts arbitrary grids; if we hard-require uniform, say so
      loudly in the docstring ("unlike MATLAB's invfreqz ...").
- [ ] **Complex/whole-spectrum designs**: `fast_equation_error_filter_design` has a
      partially-supported `is_complex` path (`omega[0] < 0`). Either finish + test it or
      cut it from v1 and validate against it.
- [ ] **`stabilize` semantics** (now: default `None` → False for direct, True for
      iterative). Confirm this is the intended contract and document that stabilization
      changes phase (magnitude-preserving pole reflection).
- [ ] **`min_phase` semantics**: it currently converts `H` via
      `min_phase_half_spectrum(H, 4*(len(H)-1))`. Decide whether the FFT size heuristic
      is caller-controllable, and whether `min_phase` belongs inside `invfreqz` at all
      vs. documenting a pre-processing recipe (there is overlap with
      `scipy.signal.minimum_phase`).
- [ ] **Cut or keep `lr0`** (learning-rate ramp on SM iterations). It is nonstandard,
      un-referenced in the literature, and interacts with `exp_window`. Recommend cutting
      from v1 and keeping in research notes.
- [ ] **`zero_clip`**: currently accepted and forwarded but unused inside the SM loop
      (the clipped-inverse helpers are not called on the active path). Wire it up or cut it.
- [ ] **Convergence/diagnostics output**: decide the v1 return signature. Options:
      plain `(b, a)`, or `(b, a)` plus an optional info dict (n_iterations, final
      coefficient-change norm, converged flag). SciPy precedent favors keeping it simple
      or gating extras behind `full_output=False`-style flags.

## 2. Numerical work

- [ ] **Ill-conditioned normal equations** (documented breadcrumb at the `solve(A, b)`
      call): for near-allpass targets the block Toeplitz system reaches rcond ~1e-18.
      Choose and implement one before submission:
      - `scipy.linalg.lstsq` (SVD) on the block system, or
      - Tikhonov regularization `solve(A + lam*I, b)` with documented `lam`, or
      - column equilibration.
      Add a regression test exercising a near-allpass target (the moForte string
      loop-filter case).
- [ ] **Verify the SM grid fix end-to-end**: `freqz(..., worN=w)` now evaluates 1/A on
      the same inclusive dc..π grid as `H`/`U`. Re-run the full 34-test driver and
      compare `total_error` before/after; tests with structure near Nyquist should
      improve or stay equal. Record the numbers in the PR description.
- [ ] **Time-aliasing of the correlation method**: small `n_spec` biases the ifft-based
      autocorrelations (visible as coefficient drift across SM iterations even for
      model-complete targets). Quantify, and document the "n_spec = Nfft/2+1, Nfft a
      power of 2, sufficiently large" guidance with an error bound or rule of thumb.
- [ ] **`check_real` tolerance policy**: silent imaginary-part discarding must become
      either an assertion with tolerance or a documented internal invariant.

## 3. Code consolidation into `_filter_design.py`

- [ ] Move `invfreqz` + private helpers (`_fast_equation_error_filter_design`,
      `_fast_steiglitz_mcbride_filter_design`, `_toeplitz_circulant_window`,
      `_invert_unstable_roots`, `_append_flip_conjugate`, min-phase helper if kept, ...)
      into `scipy/signal/_filter_design.py` with leading underscores; only `invfreqz`
      is public.
- [ ] Add `'invfreqz'` to `__all__` in `_filter_design.py` (it re-exports through
      `scipy/signal/__init__.py` automatically) and to the *Filter design* section of the
      `scipy/signal/__init__.py` module docstring (that docstring **is** the rendered API
      reference -- missing entries fail the doc build's autosummary check).
- [ ] No `meson.build` change needed if everything lands inside existing installed files
      (preferred). A new `.py` module would require a `meson.build` edit -- avoid.
- [ ] **Remove all library-code I/O**: no `print()`, no matplotlib imports, no plotting,
      no `__main__` demo in `_filter_design.py`. Convergence chatter → nothing, or
      `warnings.warn(..., stacklevel=2)` only for actionable conditions (e.g. SM hit
      `n_iter` without reaching `tol_iter`).
- [ ] Replace `assert` statements with `ValueError`/`RuntimeError` (asserts vanish under
      `python -O` and are not acceptable input validation upstream).
- [ ] Input validation up front: `np.asarray`, shape/length checks of `H` vs `U` vs `w`,
      `n_zeros/n_poles >= 0` ints, `n_iter >= 0`, finite-values check (`xp` conversion /
      array API compatibility to whatever degree `signal` currently requires -- check what
      neighbors like `freqz` do today and match).
- [ ] Type hints per current `_filter_design.py` style (match neighbors; mypy config is
      pinned -- run `python dev.py mypy`).
- [ ] Keep the `invert_unstable_roots` return contract but rename tuple members
      clearly (`(a_stable, roots, was_stable)`) -- the double-unpack bug happened because
      the third slot reads like the first.

## 4. Tests (`scipy/signal/tests/test_filter_design.py`)

Convert the demo driver into deterministic, assertion-based pytest tests -- no plots, no
prints, no accumulated `total_error`. Suggested class `TestInvfreqz` covering:

- [ ] **Exact recovery (model-complete)**: design from `freqz(b, a)` of known
      Butterworth / Chebyshev / elliptic filters; `assert_allclose(bh, b)`,
      `assert_allclose(ah, a)` at tight tolerance, over a few orders and `n_spec` sizes.
- [ ] **Round-trip property**: `freqz(invfreqz(H)) ≈ H` at looser tolerance for
      reduced-order fits (response-space, not coefficient-space).
- [ ] **Weighted design**: `U` weighting changes the fit as documented.
- [ ] **Steiglitz-McBride**: converges (norm-change below `tol_iter` within `n_iter`),
      improves output error vs. plain equation error on a reduced-order target, honors
      `b_0`/`a_0` warm start (warm-started run ≠ cold run at iteration 1; converges).
- [ ] **stabilize**: an unstable equation-error solution (max-phase target) comes back
      with all poles inside the unit circle and unchanged magnitude response
      (`assert_allclose(|freqz|)`).
- [ ] **min_phase**: min-phase conversion path produces poles/zeros inside unit circle,
      magnitude preserved within tolerance.
- [ ] **Error paths, fail fast**: unknown `method` → `ValueError`; wrong grid (last
      element ≠ π, bad omega range) → `ValueError`; mismatched `len(U) != len(H)` →
      `ValueError`; every removed/unimplemented kwarg absent from the signature.
- [ ] **Edge cases**: `n_poles=0` (FIR / all-zero fit), `n_zeros=0` (all-pole),
      `n_iter=1`, tiny `n_spec`, dc-only-ish responses, float32 input promotion.
- [ ] **Ill-conditioning regression**: near-allpass target solves without garbage
      (post-fix for item 2a), with a tolerance assertion on the response error.
- [ ] Mark anything slow with `@pytest.mark.slow` (`n_spec = 1025` cases); keep the
      default-suite additions fast (<~1 s total).
- [ ] Run: `python dev.py test -t scipy/signal/tests/test_filter_design.py -- -k invfreqz`.

## 5. Documentation

- [ ] **numpydoc docstring** in final form: one-line summary; Parameters / Returns /
      Raises / See Also / Notes / References / Examples sections.
      - `See Also`: `freqz`, `minimum_phase`, `remez`/`firls` (design counterparts),
        MATLAB-compat note.
      - `Notes`: the FFT-based equation-error formulation, the equation-error vs.
        output-error distinction, SM iteration, the uniform-grid requirement, the
        `Nfft/2+1` efficiency note, stabilization behavior, conditioning caveats.
      - `References`: [1] J. O. Smith, *Introduction to Digital Filters with Audio
        Applications*, FFT-based equation-error section (ccrma URL); [2] Steiglitz &
        McBride, "A technique for the identification of linear systems," IEEE TAC, 1965;
        [3] Levi, "Complex-curve fitting," IRE TAC, 1959 (as appropriate).
      - `Examples`: a runnable doctest (design a filter from a `freqz` response, show
        recovered coefficients) -- must pass `python dev.py refguide-check` /
        `python dev.py smoke-docs`.
- [ ] **Fix `.. versionadded::`** -- currently `1.14.2` in the scratch docstring; this
      tree is `1.15.0.dev0` and by submission time the target will be whatever release is
      next when the PR merges. Set it to the actual upcoming release at PR time.
- [ ] Release-note entry per current upstream practice (maintainers compile notes from
      PR titles/labels -- make the PR title descriptive: `ENH: signal.invfreqz: ...`).

## 6. Quality gates (all must pass locally before pushing)

- [ ] `python dev.py lint` (ruff + cython-lint) -- also `python tools/lint.py --diff-against=main`
      (what CI runs).
- [ ] `python dev.py mypy`.
- [ ] `python dev.py test -s signal` (full submodule, not just the new tests).
- [ ] `python dev.py refguide-check` / `python dev.py smoke-docs` for the new docstring.
- [ ] Optional but appreciated: an `asv` benchmark (`benchmarks/benchmarks/`) for design
      cost vs. `n_spec`/order, since the selling point is the FFT-based speed
      (`python dev.py bench -t signal`).

## 7. PR mechanics

- [ ] Branch fresh off upstream `main` (e.g. `enh-signal-invfreqz`); apply the final
      consolidated change; verify none of the `*_jos.py` scratch, TODOs, or CLAUDE.md
      changes leak in (`git diff --stat upstream/main` should list only
      `_filter_design.py`, `__init__.py` docstring, `tests/test_filter_design.py`,
      and optional benchmark file).
- [ ] PR title `ENH: signal.invfreqz: FFT-based equation-error and Steiglitz-McBride
      IIR filter design from frequency response`, body: motivation, link to the
      proposal issue + JOS reference derivation, algorithm summary, MATLAB-compat
      notes, before/after conditioning discussion, and the 34-test-driver evidence.
- [ ] Expect and budget for API-review iterations (parameter names, defaults, and the
      "trimmed options" decisions in §1 are the likely discussion points).

---

## Appendix: state as of 2026-07-02 (jos branch, uncommitted fixes applied)

Working scratch: `invfreqz_jos.py` (impl), `test_invfreqz_jos.py` (34-test demo driver),
support modules `spectrum/filter/array/*_utilities_jos.py`. A 37-agent code review
(2026-07-02) found 10 confirmed defects; all fixed in the working tree:

1. dispatcher crashed on default path (bad kwargs) -- fixed
2. default `method_iter` returned None -- default now `steiglitz_mcbride`, stubs raise
   `NotImplementedError`
3. SM prefilter evaluated 1/A on freqz's endpoint-exclusive grid -- now `worN=w`
4. `b_0`/`a_0`/`zero_clip` silently dropped -- now forwarded
5. `stabilize` docstring/default mismatch -- now `None` → `n_iter != 0`
6. `min_phase=True` was a no-op -- now converts via `min_phase_half_spectrum`
7. `ah_stable, roots, ah_stable = ...` double-unpack discarded stabilized coeffs -- fixed
   in two files
8. dangling f-strings truncated assert/labels (8 sites) -- fixed, AST-verified clean
9. unknown method silently fell back to equation_error -- now `ValueError`
10. `weight` accepted but ignored -- now `NotImplementedError`

A second workflow review of these fixes is pending; fold its findings into §1-§3.

---

> give me your honest opinion on the outstanding API decisions, considering all relevant best practices.

My honest opinion on the outstanding API decisions

The governing principle, and it's worth stating because it cuts
against how research code naturally grows: everything you cut from v1
is a future non-breaking addition; everything you ship is
forever. SciPy's deprecation cycle is two releases minimum and
reviewers know it, so they will push hardest on exactly the parameters
you're least sure about. Ship the smallest API you can defend
completely. Concretely:

1. Drop method and method_iter entirely. With prony, pade_prony, and
   gauss_newton unimplemented, these enums select among one option
   each. n_iter=0 → equation error, n_iter>0 → Steiglitz-McBride is
   the whole story, and it's exactly MATLAB's shape (their iter
   argument switches least-squares to iterative refinement -- no
   method enum). If gauss_newton ever lands, adding method= then is
   backward-compatible. Shipping NotImplementedError stubs now just
   documents promises you haven't kept.

2. Rename toward scipy house style: n_iter → maxiter, tol_iter →
   tol. maxiter/tol is the established convention across
   scipy.optimize and scipy.sparse.linalg. Reviewers will ask;
   conceding in advance costs nothing.

3. The grid contract is the one real design decision. The FFT-based
   method fundamentally requires uniform dc..π inclusive sampling --
   that's not a limitation to apologize for, it's why the method is
   fast. But MATLAB's invfreqz accepts arbitrary grids, and users
   arriving with measured responses on log grids will hit this
   immediately. My recommendation for v1: accept an optional w and
   validate it against the required grid (this catches
   Hz-vs-rad/sample unit mistakes cheaply -- a real failure mode),
   raise ValueError otherwise, and say plainly in the Notes: "unlike
   MATLAB's invfreqz, this function requires a uniform grid; here's
   why, here's what to do." Be prepared for reviewers to ask for a
   direct (non-FFT) Levi-style solve for arbitrary grids as a
   follow-up -- that's a reasonable v2, not a v1 blocker. Cut the
   half-finished complex/whole-spectrum is_complex path from v1
   outright.

4. Keep U, but document the weighting identity. I went back and forth
   here. A filter-design user thinks "weight W(ω)"; your U is
   system-ID framing. But U is strictly more general (complex input
   spectra, true system ID), it matches your published derivation, and
   the bridge is one documented line: to weight the fit by W(ω), pass
   U = sqrt(W) (since equation error minimizes Σ|U|²·|A·H − B|²). One
   parameter, both audiences served. Do not also add a weight
   parameter -- the current NotImplementedError stub should just be
   deleted.

5. Kill the dynamic stabilize default -- I implemented it to match
   your docstring, but it's the wrong design. Mode-dependent defaults
   are an API smell, and this one conflates two different things. The
   SM iteration needs its prefilter poles stable as an algorithmic
   requirement -- that shouldn't be user-visible at all; make it
   unconditional inside the loop. Whether the returned filter gets its
   poles reflected is a genuine user decision (it preserves magnitude
   but changes phase -- sometimes exactly wrong, e.g. when the target
   really is max-phase). So: internal prefilter stabilization always
   on, public stabilize=False always, applied only to the returned (b,
   a), with the phase caveat documented.

6. Cut min_phase. This is my strongest opinion. It's preprocessing of
   the target, not part of the design algorithm; it collides in name
   and concept with the existing scipy.signal.minimum_phase; and it
   drags in the n_fft knob that round 2 flagged as unactionable
   through this API. A docstring Example showing "convert H to minimum
   phase, then call invfreqz" delivers the same capability with zero
   API surface.

7. Cut lr0 (and exp_window from the public path). The learning-rate
   ramp is unreferenced in the SM literature, and reviewers will ask
   for a citation you don't have. Driver test 2's error growing across
   iterations under lr0=0.1 suggests it's not even clearly
   helping. Keep it on the jos branch as a research knob.

8. Cut zero_clip, b_0/a_0, verbose, debug. zero_clip guards nothing
   (dead code). Warm start is mly a prefilter seed -- SM's cold first
   iteration is the equation-error solve, so the
   eqn-error→SMrefinement workflow happens automatically; external
   initializations are a rare need, easily re-added. verbose/debug
   printing and plotting cannot exist in scipy library code anyway;
   the only survivor should be a warnings.warn when SM exhausts
   maxiter without reaching tol.

9. Solve with lstsq, not solve. The near-allpass rcond ~1e-18
   breadcrumb decides this. The blockoles+1) -- tiny; the FFT is where
   the speed lives, so SVD robustness at the solve step
   costsessentially nothing. lstsq also degrades gracefully on rank
   deficiency instead of gambling. Tikhonov with a magic λ would be a
   parameter you'd have to defend; equilibration is more code for the
   same outcome.

10. Return plain (b, a). No info dict, no full_output. That's signal
    namespace convention (butter, cheby1, firls all return arrays),
    and convergence trouble is a warning, not a return value.

Putting it together, the v1 I'd defend in the PR:

def invfreqz(H, n_zeros, n_poles, *, U=None, w=None,
             maxiter=0, tol=1e-8, stabilize=False):
    """... returns (b, a) ..."""

Eight names, every one implemented, every one with a one-sentence
justification. That's a signature a scipy reviewer can approve in one
pass -- and every cut above can return later as a non-breaking
enhancement once there's a user asking for it, which is the strongest
position to add API from.

The two places I'd expect genuine reviewer debate, where you should
have positions ready rather riction versus MATLAB parity (§3 --
someone will ask for arbitrary-grid support; the answer is "v2, via
the direct solve"), and U-versus-weights framing (§4 -- hold the line,
it's more general and it's your published formulation).  Stop says:
Plan active in this session -- update the plan file if status changed.
