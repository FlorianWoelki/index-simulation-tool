use ordered_float::OrderedFloat;

/// Represents a neighbor node in the HNSW graph structure.
/// Each neighbor has an associated `id` indicating its position in the dataset
/// and a `distance` which is a floating-point representation of the distance
/// from a reference point, ensuring order by distance.
#[derive(Clone, PartialEq, Debug)]
pub(super) struct NeighborNode {
    pub id: usize,
    pub distance: OrderedFloat<f64>,
}

impl NeighborNode {
    /// Constructs a new `NeighborNode`.
    ///
    /// # Arguments
    ///
    /// * `id` - A unique identifier for the neighbor node.
    /// * `distance` - The distance from the reference point to this neighbor.
    pub(super) fn new(id: usize, distance: f64) -> Self {
        NeighborNode {
            id,
            distance: OrderedFloat(distance),
        }
    }
}

impl Ord for NeighborNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}

impl PartialOrd for NeighborNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for NeighborNode {}
