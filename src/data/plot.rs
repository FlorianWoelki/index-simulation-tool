use ordered_float::OrderedFloat;
use plotly::{
    common::{DashType, Font, Marker, Title},
    layout::{Annotation, Axis, Legend, Shape, ShapeLine, ShapeType},
    Histogram, Layout, Plot,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::index::DistanceMetric;

use super::vector::SparseVector;

fn mean(data: &[f32]) -> Option<f32> {
    if data.is_empty() {
        return None;
    }

    let sum: f32 = data.iter().sum();
    Some(sum / data.len() as f32)
}

fn median(data: &[f32]) -> Option<f32> {
    if data.is_empty() {
        return None;
    }

    let mut data = data.to_owned();
    let mid = data.len() / 2;
    data.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });

    if data.len() % 2 == 0 {
        Some((data[mid - 1] + data[mid]) / 2.0)
    } else {
        Some(data[mid])
    }
}

/// Creates a histogram showing the distribution of non-zero elements in
/// a set of sparse vectors.
/// This function helps to understand the sparsity characteristics of the
/// provided dataset.
pub fn plot_sparsity_distribution(data: &[SparseVector], metadata: &str) -> Plot {
    let non_zero_counts: Vec<f32> = data.iter().map(|v| v.indices.len() as f32).collect();

    let trace = Histogram::new(non_zero_counts.clone())
        .marker(Marker::new().color("lightgray"))
        .name("Sparsity");

    let mean_value = mean(&non_zero_counts).unwrap_or(0.0);
    let median_value = median(&non_zero_counts).unwrap_or(0.0);

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(
        Layout::new()
            .title(Title::new(
                format!("Distribution of Non-Zero Elements ({})", metadata).as_str(),
            ))
            .x_axis(Axis::new().title(Title::new("Number of Non-Zero Elements")))
            .y_axis(Axis::new().title(Title::new("Frequency")))
            .legend(Legend::new().title(Title::new("Legend")))
            .shapes(vec![
                Shape::new()
                    .shape_type(ShapeType::Line)
                    .x0(mean_value)
                    .x1(mean_value)
                    .y0(0)
                    .y1(1)
                    .y_ref("paper")
                    .line(ShapeLine::new().color("red").dash(DashType::Dash)),
                Shape::new()
                    .shape_type(ShapeType::Line)
                    .x0(median_value)
                    .x1(median_value)
                    .y0(0)
                    .y1(1)
                    .y_ref("paper")
                    .line(ShapeLine::new().color("green").dash(DashType::Dash)),
            ])
            .annotations(vec![
                Annotation::new()
                    .x(mean_value)
                    .y(1)
                    .y_ref("paper")
                    .text("Mean")
                    .show_arrow(false)
                    .font(Font::new().color("red")),
                Annotation::new()
                    .x(median_value)
                    .y(0.9)
                    .y_ref("paper")
                    .text("Median")
                    .show_arrow(false)
                    .font(Font::new().color("green")),
            ]),
    );
    plot
}

/// Generates a histogram of distances between query vectors and their
/// nearest neighbors in the groundtruth data.
/// Helps you to analyze the distribution of distances in nearest neighbor
/// searches.
pub fn plot_nearest_neighbor_distances(
    query_vectors: &[SparseVector],
    groundtruth: &[SparseVector],
    metric: &DistanceMetric,
) -> Plot {
    let distances = query_vectors
        .par_iter()
        .map(|v| {
            groundtruth
                .par_iter()
                .map(|gt| OrderedFloat(v.distance(gt, metric)))
                .min()
                .unwrap()
        })
        .map(|v| v.into_inner())
        .collect::<Vec<f32>>();

    let trace = Histogram::new(distances.clone())
        .marker(Marker::new().color("lightgray"))
        .name("Distances");

    let mean_distance = mean(&distances).unwrap_or(0.0);
    let median_distance = median(&distances).unwrap_or(0.0);

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(
        Layout::new()
            .title(Title::new("Distribution of Nearest Neighbor Distances"))
            .x_axis(Axis::new().title(Title::new("Distance")))
            .y_axis(Axis::new().title(Title::new("Frequency")))
            .legend(Legend::new().title(Title::new("Legend")))
            .shapes(vec![
                Shape::new()
                    .shape_type(ShapeType::Line)
                    .x0(mean_distance)
                    .x1(mean_distance)
                    .y0(0)
                    .y1(1)
                    .y_ref("paper")
                    .line(ShapeLine::new().color("red").dash(DashType::Dash)),
                Shape::new()
                    .shape_type(ShapeType::Line)
                    .x0(median_distance)
                    .x1(median_distance)
                    .y0(0)
                    .y1(1)
                    .y_ref("paper")
                    .line(ShapeLine::new().color("green").dash(DashType::Dash)),
            ])
            .annotations(vec![
                Annotation::new()
                    .x(mean_distance)
                    .y(1)
                    .y_ref("paper")
                    .text("Mean")
                    .show_arrow(false)
                    .font(Font::new().color("red")),
                Annotation::new()
                    .x(median_distance)
                    .y(0.9)
                    .y_ref("paper")
                    .text("Median")
                    .show_arrow(false)
                    .font(Font::new().color("green")),
            ]),
    );
    plot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_empty() {
        let data: Vec<f32> = vec![];
        assert_eq!(mean(&data), None);
    }

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&data), Some(3.0));
    }

    #[test]
    fn test_median_empty() {
        let data: Vec<f32> = vec![];
        assert_eq!(median(&data), None);
    }

    #[test]
    fn test_median() {
        let data = vec![4.0, 1.0, 3.0, 2.0];
        assert_eq!(median(&data), Some(2.5));
    }
}
