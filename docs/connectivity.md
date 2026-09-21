# Connectivity before tracing

`spong.connectivity.determine_connectivity(model, enumeration=None, policy=None)`
is an experimental attaching-map stage. It checks A>0 and Morse (or accepts an
existing enumeration from the **same unchanged model**), builds the merge tree,
and classifies each of the two unstable branches of each saddle. It does not
build a portrait. Local stubs and floating section proposals remain necessary
inputs to validated transport.

```python
from spong.connectivity import determine_connectivity, ConnectivityPolicy
result = determine_connectivity(model, policy=ConnectivityPolicy(
    rheostat_diagnostics=True))
print(result.as_dict())
```

The result retains the enumeration with reusable validated launches. Critical
indices refer to `result.enumeration.points`. Branches are identified by source
index and local orientation; `direction` is their initial b direction, not a
claim of global monotonicity in b.

## Decisions and evidence

1. A validated local cone identifies a component just below the source loss.
   Its exact critical inventory forces capture when the component is bounded
   and has one minimum and no saddle. A critical-free component with just one
   unbounded end forces escape in that direction.
2. For ambiguous inventories, short floating trajectories propose crossings of
   regular sections below merge-tree levels. Existing native rational trapping
   tubes must validate the crossing. Exact root counts place the entire
   terminal enclosure in a component, whose inventory can force the destination.
   A farther Frobenius local section is attempted only for ambiguous branches.
3. Remaining saddle candidates are pruned by the certified loss ordering and
   the common roof of B-saddles. Optional rheostat calculations report section
   gaps, sensitivities, and local numerical wall candidates for these pairs.
   They cannot change a proof verdict.

`complete` means every unstable destination was certified; it does not describe
all incoming stable branches or a compactification at infinity. A complete
attaching map certifies finite-plane Smale. `finite_plane_smale` can also be true
before the attaching map is complete if all saddle destinations are excluded
but a capture/escape choice remains unresolved.

`proof` on each resolved branch retains its launch or tube witness. JSON output
is a summary, not a standalone replayable certificate. All critical-value signs
must be resolved before using a component's inventory: undecided signs must not
silently remove possible destinations.

## Current limits

This is a development API, not wired into the production inspector or resolver.
It reuses the existing native proof engine and its chart and rational-budget
limitations. It reports unresolved when a chart, launch, proposal budget, or
trapping tube fails. No theorem identifies every such failure with proximity
to a saddle connection.

The optional rheostat diagnostic uses the native C99/GMP backend described
in `wall_native.md`. It searches locally around Lambda=1, defaults to
64 GL8 steps, and caps the number of candidate pairs. It is not a complete wall
search and is not the converged 512/1024-step protocol described in
`wall_continuation.md`. A located wall's distance is an estimate, not a certified
lower bound on distance to the nearest wall. The latter is explicitly `None`.
It requires validated exclusion for every candidate throughout a parameter
interval, including branch continuation and section existence on that interval.
No compactified Smale conclusion is claimed.

The intended mature stopping rule is: prove a signed gap enclosure excludes
zero for each candidate at Lambda=1; otherwise refine the section and parameter
enclosures. For a simple wall, parameter uncertainty scales approximately as
section-gap uncertainty divided by the magnitude of the gap sensitivity.
Few-ulp root localization is therefore a useful target, not a substitute for
validated error bounds or a guarantee near tangencies.

## Checked examples

With uniform moments on [0,1]:

- f=1+x, g=1+x²: both branches reach their respective minima directly from
  launch-component inventories, with no trajectory transport.
- f=1, g=1+x: left branch escapes; right branch is captured after one validated
  decision section. Smale exclusion is already available before that transport.
- f=g=1+x+x²: three of four destinations follow from launches. The remaining
  branch currently encounters a native chart/handoff limitation and is reported
  unresolved; both stable branches of its possible lower saddle remain listed.

Tests also cover missing unstable branches, unresolved critical-value signs,
transport-budget exhaustion, and numerical rheostat data never promoting a
certificate.
