use ordered_float::OrderedFloat;

#[derive(Clone, PartialEq, Debug)]
pub struct NeighborNode {
    pub id: usize,
    pub distance: OrderedFloat<f32>,
}

impl NeighborNode {
    pub(super) fn new(id: usize, distance: f32) -> Self {
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
