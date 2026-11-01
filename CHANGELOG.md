# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0

Initial release.

### Added

- `Tropical` semiring interface with `zero`, `one`, `add`, `mul` and an explicit
  divergence predicate.
- `MaxPlus` and `MinPlus` carriers over `f64`, rejecting `NaN`.
- Dense `Matrix` with idempotent addition, tropical multiplication, powers by
  repeated squaring, and Kleene closure.
- `ClosureError::PositiveCycle` for a diverging max-plus closure and the
  corresponding negative-cycle case for min-plus.
