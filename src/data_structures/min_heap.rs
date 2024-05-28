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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_len() {
        let mut heap = MinHeap::new();
        assert_eq!(heap.len(), 0);

        heap.push(1, OrderedFloat(1.0));
        assert_eq!(heap.len(), 1);

        heap.push(2, OrderedFloat(2.0));
        assert_eq!(heap.len(), 2);
    }

    #[test]
    fn test_pop() {
        let mut heap = MinHeap::new();
        heap.push(1, OrderedFloat(1.0));
        heap.push(2, OrderedFloat(2.0));
        heap.push(3, OrderedFloat(0.5));

        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), Some(2));
    }

    #[test]
    fn test_peek() {
        let mut heap = MinHeap::new();
        assert_eq!(heap.peek(), None);

        heap.push(1, OrderedFloat(1.0));
        heap.push(2, OrderedFloat(2.0));
        assert_eq!(heap.peek(), Some(&1));

        heap.push(3, OrderedFloat(0.0));
        assert_eq!(heap.peek(), Some(&3));
    }

    #[test]
    fn test_into_sorted_vec() {
        let mut heap = MinHeap::new();
        heap.push(1, OrderedFloat(1.0));
        heap.push(2, OrderedFloat(2.0));
        heap.push(3, OrderedFloat(0.5));

        assert_eq!(heap.into_sorted_vec(), vec![2, 1, 3]);
    }
}
