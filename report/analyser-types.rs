use tfpb::types::DataType;
use Tensor;
use Result;

/// Partial information about any value.
pub trait Fact: Clone + PartialEq {
    type Concrete;

    /// Tries to transform the fact into a concrete value.
    fn concretize(&self) -> Option<Self::Concrete>;

    /// Returns whether the value is fully determined.
    fn is_concrete(&self) -> bool {
        self.concretize().is_some()
    }

    /// Tries to unify the fact with another fact of the same type.
    fn unify(&self, other: &Self) -> Result<Self>;
}

/// Partial information about a tensor.
///
/// The task of the analyser is to tag every edge in the graph with information
/// about the tensors that flow through it - specifically their datatype, their
/// shape and possibly their value. During the analysis, however, we might only
/// know some of that information (say, for instance, that an edge only carries
/// tensors of rank 4, but without knowing their precise dimension).
///
/// This is where tensor facts come in: they hold partial information about the
/// datatype, shape and value of tensors that might flow through an edge of the
/// graph. The analyser will first tag each edge with a fact, starting with the
/// most general one and specializing it at each iteration. Eventually, it will
/// reach a fixed point that - hopefully - holds enough information.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Clone, PartialEq)]
pub struct TensorFact {
    pub datatype: TypeFact,
    pub shape: ShapeFact,
    pub value: ValueFact,
}

impl Fact for TensorFact {
    type Concrete = Tensor;

    /// Tries to transform the fact into a concrete value.
    fn concretize(&self) -> Option<Self::Concrete> {
        self.value.concretize()
    }

    /// Tries to unify the fact with another fact of the same type.
    fn unify(&self, other: &Self) -> Result<Self> {
        let tensor = TensorFact {
            datatype: self.datatype.unify(&other.datatype)?,
            shape:    self.shape.unify(&other.shape)?,
            value:    self.value.unify(&other.value)?,
        };

        trace!("Unifying {:?} with {:?} into {:?}.", self, other, tensor);

        Ok(tensor)
    }
}

/// Partial information about a value of type T.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Clone, PartialEq)]
pub enum GenericFact<Clone + PartialEq> {
    Any,
    Only(T)
}

impl<T: Clone + PartialEq> Fact for GenericFact<T> {
    type Concrete = T;

    /// Tries to transform the fact into a concrete value.
    fn concretize(&self) -> Option<T> {
        match self {
            GenericFact::Any => None,
            GenericFact::Only(m) => Some(m.clone()),
        }
    }

    /// Tries to unify the fact with another fact of the same type.
    fn unify(&self, other: &Self) -> Result<Self> {
        let fact = match (self, other) {
            (_, GenericFact::Any) => self.clone(),
            (GenericFact::Any, _) => other.clone(),
            _ if self == other    => self.clone(),
            _ => bail!("Impossible to unify {:?} with {:?}.", self, other),
        };

        Ok(fact)
    }
}

/// Partial information about a type.
pub type TypeFact = GenericFact<DataType>;

/// Partial information about a shape.
///
/// A basic example of a shape fact is `shapefact![1, 2]`, which corresponds to
/// the shape `[1, 2]` in Tensorflow. We can use `_` in facts to denote unknown
/// dimensions (e.g. `shapefact![1, 2, _]` corresponds to any shape `[1, 2, k]`
/// with `k` a non-negative integer). We can also use `..` at the end of a fact
/// to only specify its first dimensions, so `shapefact![1, 2; ..]` matches any
/// shape that starts with `[1, 2]` (e.g. `[1, 2, i]` or `[1, 2, i, j]`), while
/// `shapefact![..]` matches any shape.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Clone, PartialEq)]
pub struct ShapeFact {
    pub open: bool,
    pub dims: Vec<DimFact>,
}

impl ShapeFact {
    /// Constructs an open shape fact.
    pub fn open(dims: Vec<DimFact>) -> ShapeFact {
        ShapeFact { open: true, dims }
    }

    /// Constructs a closed shape fact.
    pub fn closed(dims: Vec<DimFact>) -> ShapeFact {
        ShapeFact { open: false, dims }
    }
}

impl Fact for ShapeFact {
    type Concrete = Vec<usize>;

    /// Tries to transform the fact into a `Vec<usize>`, or returns `None`.
    fn concretize(self: &ShapeFact) -> Option<Vec<usize>> {
        if self.open {
            debug!("Impossible to concretize an open shape.");
            return None;
        }

        let dims: Vec<_> = self.dims.iter().filter_map(|d| d.concretize()).collect();

        if dims.len() < self.dims.len() {
            debug!("Impossible to concretize a shape with unknown dimensions.");
            None
        } else {
            Some(dims)
        }
    }

    /// Tries to unify the fact with another fact of the same type.
    fn unify(&self, other: &Self) -> Result<Self> {
        let (x, y) = (self, other);

        use itertools::EitherOrBoth::{Both, Left, Right};
        use itertools::Itertools;

        let xi = x.dims.iter();
        let yi = y.dims.iter();

        let dimensions: Vec<_> = xi.zip_longest(yi)
            .map(|r| match r {
                Both(a, b) => a.unify(b),
                Left(d) if y.open => Ok(*d),
                Right(d) if x.open => Ok(*d),

                Left(_) | Right(_) => bail!(
                    "Impossible to unify closed shapes of different rank (found {:?} and {:?}).",
                    x,
                    y
                ),
            })
            .collect::<Result<_>>()?;

        if x.open && y.open {
            Ok(ShapeFact::open(dimensions))
        } else {
            Ok(ShapeFact::closed(dimensions))
        }
    }
}

/// Partial information about a dimension.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Clone, Copy, PartialEq)]
pub enum DimFact {
    Any,
    Streamed,
    Only(usize),
}

impl DimFact {
    /// Returns whether the dimension is streamed.
    pub fn is_streamed(&self) -> bool {
        self == &DimFact::Streamed
    }
}

impl Fact for DimFact {
    type Concrete = usize;

    /// Tries to transform the dimension fact into an `usize`, or returns `None`.
    fn concretize(&self) -> Option<usize> {
        match self {
            DimFact::Any => None,
            DimFact::Streamed => None,
            DimFact::Only(i) => Some(*i),
        }
    }

    /// Returns whether the dimension is concrete.
    fn is_concrete(&self) -> bool {
        match self {
            DimFact::Any => false,
            DimFact::Streamed => true,
            DimFact::Only(_) => true,
        }
    }

    /// Tries to unify the fact with another `DimFact`.
    fn unify(&self, other: &Self) -> Result<Self> {
        let fact = match (self, other) {
            (_, DimFact::Any)  => self.clone(),
            (DimFact::Any, _)  => other.clone(),
            _ if self == other => self.clone(),
            _ => bail!("Impossible to unify {:?} with {:?}.", self, other),
        };

        Ok(fact)
    }
}

/// Partial information about a value.
pub type ValueFact = GenericFact<Tensor>;