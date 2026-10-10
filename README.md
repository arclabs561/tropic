# tropic

Local draft spot. Not a crate yet.

Purpose: tropical and max-plus algebra: max-plus matrices, tropical matrix
multiply, Kleene star, shortest-path duals, tropical eigenvectors, Hilbert
metric, and small polytope helpers.

Crates.io exact-name check with `cargo info tropic`: no crate found on
2026-06-26.

First proof: max-plus matrix multiply plus either a shortest-path example or a
second tropical-attention score example.

Extraction rule: start as the smallest primitive needed by a real example, not
as a tropical neural-network or algebraic-geometry package.

Current status: defer. The `ricci` and `hopfield` examples both use max-plus
path bias, but neither needs general tropical algebra yet. If a third example
needs reusable max-plus matrix operations independent of score-bias composition,
revisit this boundary.

Boundary with `attbias`: score-bias composition is not tropical algebra.
If the only repeated operation is "add max-plus-derived bias before sparsemax
or attention," keep it in `attbias` or example-local code. `tropic` should only
exist for reusable max-plus/tropical operations independent of that score-bias
surface.
