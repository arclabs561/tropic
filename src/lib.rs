//! Tropical (max-plus) and min-plus algebra over the extended reals.
//!
//! The tropical semiring replaces `+` with `max` (or `min`) and `*` with `+`.
//! It is idempotent (`a ⊕ a = a`), which is what makes it the natural home for
//! longest-path, shortest-path, dynamic-programming and path-bias work.
//!
//! ```
//! use tropic::{Matrix, MaxPlus};
//!
//! // Two hops of a 3-node path graph: 0 -> 1 -> 2.
//! let step = Matrix::from_rows(vec![
//!     vec![MaxPlus::from(0.0), MaxPlus::from(1.0), MaxPlus::neg_inf()],
//!     vec![MaxPlus::neg_inf(), MaxPlus::from(0.0), MaxPlus::from(2.0)],
//!     vec![MaxPlus::neg_inf(), MaxPlus::neg_inf(), MaxPlus::from(0.0)],
//! ]).unwrap();
//!
//! let closure = step.closure().unwrap();
//! assert_eq!(closure.get(0, 2), MaxPlus::from(3.0));
//! ```
//!
//! # Scope
//!
//! This is a primitive, not a tropical-geometry package: semiring elements,
//! dense matrices, powers and Kleene closure. `closure` reports a positive
//! cycle instead of silently diverging, because a max-plus closure is only
//! finite when no reachable cycle has positive weight.
//!
//! NaN is not a valid carrier value; construction rejects it.

#![warn(missing_docs)]

use std::fmt;

/// A tropical carrier: a value with idempotent addition and additive
/// multiplication.
///
/// Implementors must satisfy the semiring laws: associativity of `add` and
/// `mul`, distributivity of `mul` over `add`, and identities `zero` (additive)
/// and `one` (multiplicative). `add` must be idempotent.
pub trait Tropical: Copy + PartialOrd + fmt::Debug {
    /// The additive identity, the weight of an absent edge.
    fn zero() -> Self;

    /// The multiplicative identity, the weight of a self-loop.
    fn one() -> Self;

    /// Idempotent addition: the better of two weights.
    fn add(self, rhs: Self) -> Self;

    /// Multiplication: composition of weights.
    fn mul(self, rhs: Self) -> Self;

    /// Whether this value is the additive identity.
    fn is_zero(self) -> bool {
        self == Self::zero()
    }

    /// Whether a cycle with this weight makes the closure diverge.
    ///
    /// Max-plus diverges on positive cycles, min-plus on negative ones.
    fn diverges_on_cycle(self) -> bool;
}

/// A weight in the max-plus semiring: `add` is `max`, `mul` is `+`.
///
/// The additive identity is negative infinity, the multiplicative identity is
/// zero.
#[derive(Clone, Copy, Debug)]
pub struct MaxPlus(pub f64);

impl MaxPlus {
    /// The additive identity, representing "no path".
    pub const fn neg_inf() -> Self {
        Self(f64::NEG_INFINITY)
    }

    /// The carrier value.
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl From<f64> for MaxPlus {
    /// Builds a weight.
    ///
    /// # Panics
    ///
    /// Panics when `value` is NaN, which has no order and breaks the laws.
    fn from(value: f64) -> Self {
        assert!(!value.is_nan(), "NaN is not a tropical weight");
        Self(value)
    }
}

impl PartialEq for MaxPlus {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for MaxPlus {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Tropical for MaxPlus {
    fn zero() -> Self {
        Self(f64::NEG_INFINITY)
    }

    fn one() -> Self {
        Self(0.0)
    }

    fn add(self, rhs: Self) -> Self {
        Self(self.0.max(rhs.0))
    }

    fn mul(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }

    fn diverges_on_cycle(self) -> bool {
        self.0 > 0.0
    }
}

/// A weight in the min-plus semiring: `add` is `min`, `mul` is `+`.
///
/// The additive identity is positive infinity, the multiplicative identity is
/// zero. This is the shortest-path semiring.
#[derive(Clone, Copy, Debug)]
pub struct MinPlus(pub f64);

impl MinPlus {
    /// The additive identity, representing "no path".
    pub const fn pos_inf() -> Self {
        Self(f64::INFINITY)
    }

    /// The carrier value.
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl From<f64> for MinPlus {
    /// Builds a weight.
    ///
    /// # Panics
    ///
    /// Panics when `value` is NaN, which has no order and breaks the laws.
    fn from(value: f64) -> Self {
        assert!(!value.is_nan(), "NaN is not a tropical weight");
        Self(value)
    }
}

impl PartialEq for MinPlus {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for MinPlus {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Tropical for MinPlus {
    fn zero() -> Self {
        Self(f64::INFINITY)
    }

    fn one() -> Self {
        Self(0.0)
    }

    fn add(self, rhs: Self) -> Self {
        Self(self.0.min(rhs.0))
    }

    fn mul(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }

    fn diverges_on_cycle(self) -> bool {
        self.0 < 0.0
    }
}

/// Why a closure could not be computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClosureError {
    /// A reachable cycle has positive weight, so the closure diverges.
    PositiveCycle,
}

impl fmt::Display for ClosureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PositiveCycle => {
                write!(
                    f,
                    "a reachable cycle has positive weight; the closure diverges"
                )
            }
        }
    }
}

impl std::error::Error for ClosureError {}

/// Why a matrix could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixError {
    /// Rows have different lengths.
    RaggedRows,
    /// The matrix has no rows, or no columns.
    Empty,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RaggedRows => write!(f, "rows have different lengths"),
            Self::Empty => write!(f, "a matrix needs at least one row and one column"),
        }
    }
}

impl std::error::Error for MatrixError {}

/// A dense matrix over a tropical semiring.
///
/// Entry `(i, j)` is the weight of one step from `i` to `j`; `T::zero()` means
/// no direct edge.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T: Tropical> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Tropical> Matrix<T> {
    /// Builds a matrix from rows, rejecting ragged or empty input.
    pub fn from_rows(rows: Vec<Vec<T>>) -> Result<Self, MatrixError> {
        let n = rows.len();
        if n == 0 {
            return Err(MatrixError::Empty);
        }
        let cols = rows[0].len();
        if cols == 0 {
            return Err(MatrixError::Empty);
        }
        if rows.iter().any(|row| row.len() != cols) {
            return Err(MatrixError::RaggedRows);
        }
        let data = rows.into_iter().flatten().collect();
        Ok(Self {
            rows: n,
            cols,
            data,
        })
    }

    /// A matrix of the additive identity everywhere.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::zero(); rows * cols],
        }
    }

    /// The multiplicative identity.
    ///
    /// # Panics
    ///
    /// Panics when `size` is zero.
    pub fn identity(size: usize) -> Self {
        assert!(size > 0, "identity needs at least one row");
        let mut matrix = Self::zeros(size, size);
        for i in 0..size {
            matrix.set(i, i, T::one());
        }
        matrix
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Reads an entry.
    ///
    /// # Panics
    ///
    /// Panics when the index is out of bounds.
    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[row * self.cols + col]
    }

    /// Writes an entry.
    ///
    /// # Panics
    ///
    /// Panics when the index is out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.data[row * self.cols + col] = value;
    }

    /// Idempotent entrywise addition.
    ///
    /// # Panics
    ///
    /// Panics when the shapes differ.
    pub fn add(&self, rhs: &Self) -> Self {
        assert_eq!(
            (self.rows, self.cols),
            (rhs.rows, rhs.cols),
            "add needs equal shapes"
        );
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(&rhs.data)
                .map(|(a, b)| a.add(*b))
                .collect(),
        }
    }

    /// Tropical matrix multiplication.
    ///
    /// # Panics
    ///
    /// Panics when `self.cols != rhs.rows`.
    pub fn mul(&self, rhs: &Self) -> Self {
        assert_eq!(self.cols, rhs.rows, "mul needs matching inner dimension");
        let mut out = Self::zeros(self.rows, rhs.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let left = self.get(i, k);
                if left.is_zero() {
                    continue;
                }
                for j in 0..rhs.cols {
                    let right = rhs.get(k, j);
                    if right.is_zero() {
                        continue;
                    }
                    let candidate = left.mul(right);
                    let current = out.get(i, j);
                    out.set(i, j, current.add(candidate));
                }
            }
        }
        out
    }

    /// `self` raised to `exponent` by repeated squaring, with `self^0` the
    /// identity.
    pub fn pow(&self, exponent: u32) -> Self {
        assert_eq!(self.rows, self.cols, "pow needs a square matrix");
        let mut result = Self::identity(self.rows);
        let mut base = self.clone();
        let mut e = exponent;
        while e > 0 {
            if e & 1 == 1 {
                result = result.mul(&base);
            }
            e >>= 1;
            if e > 0 {
                base = base.mul(&base);
            }
        }
        result
    }

    /// The Kleene closure `I ⊕ A ⊕ A² ⊕ ...`, the all-pairs path weights
    /// including the zero-length self-path.
    ///
    /// Returns [`ClosureError::PositiveCycle`] when a reachable cycle has
    /// positive weight, because the sum then diverges.
    pub fn closure(&self) -> Result<Self, ClosureError> {
        assert_eq!(self.rows, self.cols, "closure needs a square matrix");
        let n = self.rows;
        let mut out = self.clone();
        for k in 0..n {
            if out.get(k, k).diverges_on_cycle() {
                return Err(ClosureError::PositiveCycle);
            }
            for i in 0..n {
                let via = out.get(i, k);
                if via.is_zero() {
                    continue;
                }
                for j in 0..n {
                    let suffix = out.get(k, j);
                    if suffix.is_zero() {
                        continue;
                    }
                    let candidate = via.mul(suffix);
                    let current = out.get(i, j);
                    out.set(i, j, current.add(candidate));
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max(rows: &[&[f64]]) -> Matrix<MaxPlus> {
        Matrix::from_rows(
            rows.iter()
                .map(|row| row.iter().copied().map(MaxPlus::from).collect())
                .collect(),
        )
        .unwrap()
    }

    fn min(rows: &[&[f64]]) -> Matrix<MinPlus> {
        Matrix::from_rows(
            rows.iter()
                .map(|row| row.iter().copied().map(MinPlus::from).collect())
                .collect(),
        )
        .unwrap()
    }

    fn neg_inf_rows(rows: &[&[Option<f64>]]) -> Matrix<MaxPlus> {
        Matrix::from_rows(
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|v| v.map_or(MaxPlus::neg_inf(), MaxPlus::from))
                        .collect()
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn max_plus_addition_is_max_and_idempotent() {
        let a = MaxPlus::from(2.0);
        let b = MaxPlus::from(5.0);
        assert_eq!(a.add(b), b);
        assert_eq!(a.add(a), a);
        assert_eq!(a.mul(b), MaxPlus::from(7.0));
        assert_eq!(a.add(MaxPlus::zero()), a);
        assert_eq!(a.mul(MaxPlus::one()), a);
    }

    #[test]
    fn min_plus_addition_is_min() {
        let a = MinPlus::from(2.0);
        let b = MinPlus::from(5.0);
        assert_eq!(a.add(b), a);
        assert_eq!(a.mul(b), MinPlus::from(7.0));
        assert_eq!(a.add(MinPlus::zero()), a);
    }

    #[test]
    #[should_panic(expected = "NaN")]
    fn nan_is_rejected() {
        let _ = MaxPlus::from(f64::NAN);
    }

    #[test]
    fn multiplication_composes_path_weights() {
        let a = max(&[&[0.0, 1.0], &[f64::NEG_INFINITY, 0.0]]);
        let b = max(&[&[0.0, 2.0], &[f64::NEG_INFINITY, 0.0]]);
        let product = a.mul(&b);
        assert_eq!(product.get(0, 1), MaxPlus::from(2.0));
        assert_eq!(product.get(1, 0), MaxPlus::neg_inf());
    }

    #[test]
    fn identity_and_power_agree() {
        let step = max(&[&[0.0, 1.0], &[0.0, 0.0]]);
        assert_eq!(step.mul(&Matrix::identity(2)), step);
        assert_eq!(Matrix::identity(2).mul(&step), step);
        let mut repeated = Matrix::identity(2);
        for _ in 0..3 {
            repeated = repeated.mul(&step);
        }
        assert_eq!(step.pow(3), repeated);
    }

    #[test]
    fn semiring_laws_hold_on_a_sample() {
        let a = max(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b = max(&[&[5.0, 6.0], &[7.0, 8.0]]);
        let c = max(&[&[9.0, 10.0], &[11.0, 12.0]]);
        // Associativity of multiplication.
        assert_eq!(a.mul(&b).mul(&c), a.mul(&b.mul(&c)));
        // Distributivity of multiplication over addition.
        assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
        // Idempotence of addition.
        assert_eq!(a.add(&a), a);
    }

    #[test]
    fn closure_matches_longest_paths_on_a_dag() {
        let graph = neg_inf_rows(&[
            &[Some(0.0), Some(4.0), None, Some(1.0)],
            &[None, Some(0.0), Some(2.0), None],
            &[None, None, Some(0.0), Some(3.0)],
            &[None, None, None, Some(0.0)],
        ]);
        let closure = graph.closure().unwrap();
        // 0 -> 3 directly is 1, but 0 -> 1 -> 2 -> 3 is 9.
        assert_eq!(closure.get(0, 3), MaxPlus::from(9.0));
        assert_eq!(closure.get(0, 2), MaxPlus::from(6.0));
        assert_eq!(closure.get(3, 0), MaxPlus::neg_inf());
        // The zero-length path is included.
        assert_eq!(closure.get(2, 2), MaxPlus::from(0.0));
    }

    #[test]
    fn closure_reports_a_positive_cycle() {
        let graph = max(&[&[0.0, 1.0], &[2.0, 0.0]]);
        assert_eq!(graph.closure(), Err(ClosureError::PositiveCycle));
    }

    #[test]
    fn zero_weight_cycle_is_not_a_positive_cycle() {
        // A zero-weight cycle neither diverges nor changes reachability.
        let graph = max(&[&[0.0, 0.0], &[0.0, 0.0]]);
        let closure = graph.closure().unwrap();
        assert_eq!(closure.get(0, 1), MaxPlus::from(0.0));
    }

    #[test]
    fn min_plus_closure_is_shortest_paths() {
        let graph = Matrix::from_rows(vec![
            vec![MinPlus::from(0.0), MinPlus::from(4.0), MinPlus::pos_inf()],
            vec![MinPlus::pos_inf(), MinPlus::from(0.0), MinPlus::from(2.0)],
            vec![MinPlus::from(1.0), MinPlus::pos_inf(), MinPlus::from(0.0)],
        ])
        .unwrap();
        let closure = graph.closure().unwrap();
        assert_eq!(closure.get(0, 2), MinPlus::from(6.0));
        assert_eq!(closure.get(2, 1), MinPlus::from(5.0));
    }

    #[test]
    fn max_plus_square_reproduces_the_example_copy() {
        // The 4x4 helper duplicated in ricci and hopfield examples.
        let a = max(&[
            &[0.0, 1.0, f64::NEG_INFINITY, f64::NEG_INFINITY],
            &[f64::NEG_INFINITY, 0.0, 2.0, f64::NEG_INFINITY],
            &[f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0, 3.0],
            &[f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0],
        ]);
        let squared = a.mul(&a);
        assert_eq!(squared.get(0, 2), MaxPlus::from(3.0));
        // Three hops, so a square cannot reach 0 -> 3.
        assert_eq!(squared.get(0, 3), MaxPlus::neg_inf());
        assert_eq!(squared.get(1, 3), MaxPlus::from(5.0));
    }

    #[test]
    fn ragged_and_empty_inputs_are_rejected() {
        assert_eq!(
            Matrix::from_rows(vec![vec![MaxPlus::from(1.0)], vec![]]),
            Err(MatrixError::RaggedRows)
        );
        assert_eq!(
            Matrix::<MaxPlus>::from_rows(vec![]),
            Err(MatrixError::Empty)
        );
    }

    #[test]
    fn shortest_path_toy_matches_hand_computation() {
        let graph = min(&[
            &[0.0, 1.0, f64::INFINITY, f64::INFINITY],
            &[f64::INFINITY, 0.0, 1.0, f64::INFINITY],
            &[f64::INFINITY, f64::INFINITY, 0.0, 1.0],
            &[f64::INFINITY, f64::INFINITY, f64::INFINITY, 0.0],
        ]);
        let closure = graph.closure().unwrap();
        assert_eq!(closure.get(0, 3), MinPlus::from(3.0));
        assert_eq!(closure.get(3, 0), MinPlus::pos_inf());
    }
}
