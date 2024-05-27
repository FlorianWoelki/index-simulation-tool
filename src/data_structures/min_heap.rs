use std::{cmp::Ordering, collections::BinaryHeap};

use ordered_float::OrderedFloat;

#[derive(Debug)]
pub struct MinHeap<T>(BinaryHeap<MinHeapEntry<T>>);

#[derive(Debug)]
struct MinHeapEntry<T>(T, OrderedFloat<f32>);

impl<T> MinHeapEntry<T> {
    fn new(item: T, score: OrderedFloat<f32>) -> Self {
        MinHeapEntry(item, score)
    }
}

impl<T> Eq for MinHeapEntry<T> {}

impl<T> PartialEq for MinHeapEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<T> Ord for MinHeapEntry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.1.cmp(&self.1)
    }
}

impl<T> PartialOrd for MinHeapEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> MinHeap<T> {
    pub fn new() -> Self {
        MinHeap(BinaryHeap::new())
    }

    pub fn push(&mut self, item: T, score: OrderedFloat<f32>) {
        self.0.push(MinHeapEntry::new(item, score));
    }

    pub fn pop(&mut self) -> Option<T> {
        self.0.pop().map(|entry| entry.0)
    }

    pub fn peek(&self) -> Option<&T> {
        self.0.peek().map(|entry| &entry.0)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn into_sorted_vec(self) -> Vec<T> {
        self.0
            .into_sorted_vec()
            .into_iter()
            .map(|entry| entry.0)
            .collect()
    }
}
