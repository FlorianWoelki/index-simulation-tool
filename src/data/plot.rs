use plotly::{common::Title, layout::Axis, BoxPlot, Histogram, Layout, Plot, Scatter};

use crate::{data::SparseVector, index::DistanceMetric};

pub fn plot_sparsity_distribution(data: &[SparseVector]) -> Plot {
    let non_zero_counts: Vec<usize> = data.iter().map(|v| v.indices.len()).collect();

    let trace = Histogram::new(non_zero_counts);
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(
        Layout::new()
            .title(Title::new("Distribution of Non-Zero Elements"))
            .x_axis(Axis::new().title(Title::new("Number of Non-Zero Elements")))
            .y_axis(Axis::new().title(Title::new("Frequency"))),
    );
    plot
}

pub fn plot_nearest_neighbor_distances(
    query_vectors: &[SparseVector],
    groundtruth: &[Vec<SparseVector>],
    metric: &DistanceMetric,
) -> Plot {
    let distances: Vec<f32> = query_vectors
        .iter()
        .zip(groundtruth.iter())
        .map(|(q, gt)| q.distance(&gt[0], metric))
        .collect();

    let trace = Histogram::new(distances);
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(
        plotly::Layout::new()
            .title(plotly::common::Title::new(
                "Distribution of Nearest Neighbor Distances",
            ))
            .x_axis(plotly::layout::Axis::new().title(plotly::common::Title::new("Distance")))
            .y_axis(plotly::layout::Axis::new().title(plotly::common::Title::new("Frequency"))),
    );
    plot
}
