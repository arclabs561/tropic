# tropic

Max-plus and min-plus algebra with law-tested semiring matrices and closure.

Tropical algebra replaces `+` with `max` (or `min`) and `×` with `+`. Addition
becomes idempotent (`a ⊕ a = a`), which is what makes the semiring the natural
home for longest-path, shortest-path, dynamic-programming and attention-bias
work.

```rust
use tropic::{Matrix, MaxPlus};

let step = Matrix::from_rows(vec![
    vec![MaxPlus::from(0.0), MaxPlus::from(1.0), MaxPlus::neg_inf()],
    vec![MaxPlus::neg_inf(), MaxPlus::from(0.0), MaxPlus::from(2.0)],
    vec![MaxPlus::neg_inf(), MaxPlus::neg_inf(), MaxPlus::from(0.0)],
]).unwrap();

// All-pairs path weights: 0 -> 2 is 1 + 2.
assert_eq!(step.closure().unwrap().get(0, 2), MaxPlus::from(3.0));
```

## What it is

- `Tropical`: the semiring interface (`zero`, `one`, `add`, `mul`) with an
  explicit divergence predicate, so a closure cannot silently diverge.
- `MaxPlus` and `MinPlus`: the two carriers, over `f64`. `NaN` is rejected
  because it has no order and breaks the laws.
- `Matrix`: dense semiring matrices with `add`, `mul`, `pow` and `closure`.
  `closure` returns `ClosureError::PositiveCycle` instead of an infinite answer.

## Boundary

This is a primitive, not a tropical-geometry or tropical-neural-network
package. It does not provide tropical eigenvalues, polytopes, or symbolic
tropical curves; those arrive only when a consumer needs them.

Score-bias composition is not tropical algebra. If the repeated operation is
"add a max-plus-derived bias before attention", that belongs with the attention
code, not here.

## Testing

`just check` runs formatting, Clippy with `-D warnings`, the test suite and
docs. The tests cover the semiring laws, matrix identity and power, closure
against hand-computed longest paths, positive-cycle detection, and min-plus
shortest paths.

## License

MIT OR Apache-2.0.
