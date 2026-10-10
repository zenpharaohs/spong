# Continuous-Bernoulli interior near-endpoint draw stalls

**Status: resolved in the sibling sampler repository.** With
`continuous-bernoulli` commit `c959353` and SPONG's auto-built
`libcb_core-9286ba8beed2a3ec.dylib`, the minimal draw below returns in
about 40 µs. The original 3,200-round Fisher/Gauss–Newton allocator
comparison now completes in about 2.35 s, including CB playback and
extensions. The details below preserve the original regression case.

SPONG's Fisher/Gauss–Newton allocator exposes a sampler stall at a valid,
strictly interior posterior state. Please investigate in the
`continuous-bernoulli` repository. This is a performance/termination bug;
do not round the observation to an endpoint or impose an artificial margin.

## Minimal reproduction

From the SPONG repository, with its Python environment and the sibling
`continuous-bernoulli` source checkout available:

```python
from demos import cb_sampler

library = cb_sampler.resolve_library(auto_build=True)
with cb_sampler.ContinuousBernoulliBank(1, seed=1729, library=library) as bank:
    bank.set_observation(0, 3.8758505860895066e-32, 7)
    print(bank.draw_all())  # did not return within 5 seconds
```

Run this in a child process with a timeout. `set_observation` passes
`(chi, nu) = (7 * 3.8758505860895066e-32, 7)` to `cb_stream_set_stats`.
The stall is in `cb_stream_draw`, not model construction, gradient evaluation,
or the SPONG allocator loop. On the same build, `Z=0` returns its point mass
in about 10 µs; `Z=1e-12`, `1e-6`, and `0.1` also return in about 10 µs.
`Z=1e-20` exceeds a 2-second timeout. These positive values remain inside
the exact posterior's support and must be sampled rather than clipped.

## Original SPONG trigger

Use `inverse.separated_linear_case(4, 30)` with uniform-[0,1] moments and
the skeleton-fit view
`[-0.3500000009477955, 0.35000000282441074, -39306.903837165286,
20468.486229066093]`. Run the allocator with 32 low-discrepancy starts,
seed 1729, batch size 32, chunk length 10, and `method="fisher-gn"`.
The 64-round run computes its entire uniform prefix in roughly 0.07 s and
then hangs in bandit selection. Instrumenting `ContinuousBernoulliBank`
shows arm 0 reaching `(Z, N) = (3.8758505860895066e-32, 7)` at draw 7.
The 3,200-round uniform prefix alone completes in about 1.2 s for 32,000
optimizer updates, with 40,409 minibatch passes and best exact loss
`7.5007076416611085e-06`. The sampler stall prevents the corresponding
uniform-versus-CB comparison from completing inside the interactive limit.

Observed with `continuous-bernoulli` commit `3fa5188`, SPONG base commit
`451edfb` plus the Fisher/Gauss–Newton work, Python 3.14.6 on arm64 macOS.
SPONG auto-built the content-addressed library
`libcb_core-ac731eafa9d2cc6a.dylib` from the sibling C source.

Expected: `cb_stream_draw` returns a sample promptly for finite
`0 < chi < nu` (including the near-endpoint regime), preserving the proper
continuous posterior. Please add a regression around the direct stream API
and verify repeated `set_stats`/draw transitions, since the allocator updates
the same stream after each chosen chunk.
