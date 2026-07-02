import numpy as np

from scipy.ndimage import (
    distance_transform_edt,
    label as ndi_label,
    generate_binary_structure,
    binary_dilation
)


def expand_regions(
    expanded_mask,
    prediction,
    pred_anomalous_slices
):
    for anomalous_slice in pred_anomalous_slices:

        upper_half = expanded_mask[
            anomalous_slice,
            :48,
            :
        ]

        lower_half = expanded_mask[
            anomalous_slice,
            48:,
            :
        ]

        upper_volume, upper_num = ndi_label(
            upper_half
        )

        upper_counts = np.bincount(
            upper_volume.flatten()
        )

        if len(upper_counts) > 0:
            upper_counts[0] = 0

        lower_volume, lower_num = ndi_label(
            lower_half
        )

        lower_counts = np.bincount(
            lower_volume.flatten()
        )

        if len(lower_counts) > 0:
            lower_counts[0] = 0

        if (
            upper_num > 2
            or (
                len(upper_counts) > 0
                and upper_counts.max() >= 10
            )
        ):
            expanded_mask[
                anomalous_slice,
                :48,
                :
            ] = prediction[
                anomalous_slice,
                :48,
                :
            ]

        if (
            lower_num > 2
            or (
                len(lower_counts) > 0
                and lower_counts.max() >= 10
            )
        ):
            expanded_mask[
                anomalous_slice,
                48:,
                :
            ] = prediction[
                anomalous_slice,
                48:,
                :
            ]

    clustered_prediction, num_regions = ndi_label(
        prediction
    )

    for i in range(
        1,
        num_regions + 1
    ):
        region_mask = (
            clustered_prediction == i
        ).astype(np.uint8)

        region_size = np.sum(region_mask)

        if region_size <= 4:

            structure = generate_binary_structure(
                3,
                1
            )

            dilated_region = binary_dilation(
                region_mask,
                structure=structure,
                iterations=1
            )

            expanded_mask[
                dilated_region
            ] = 1

    return expanded_mask


def expand_using_grad_cam(
    prediction,
    heatmap,
    pred_anomalous_slices,
    threshold_distance=3,
    threshold_heatmap=0.7
):
    prediction = np.asarray(
        prediction,
        dtype=np.uint8
    )

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32
    )

    distance = distance_transform_edt(
        1 - prediction
    )

    allowed_region = (
        distance <= threshold_distance
    ).astype(np.uint8)

    binary_heatmap = (
        heatmap >= threshold_heatmap
    ).astype(np.uint8)

    final_mask = np.logical_or(
        prediction,
        np.logical_and(
            binary_heatmap,
            allowed_region
        )
    ).astype(np.uint8)

    final_mask = np.logical_or(
        final_mask,
        allowed_region
    ).astype(np.uint8)

    return expand_regions(
        final_mask,
        prediction,
        pred_anomalous_slices
    )


def main(
    prediction,
    heatmap,
    pred_anomalous_slices,
    threshold_distance=3,
    threshold_heatmap=0.7
):
    return expand_using_grad_cam(
        prediction=prediction,
        heatmap=heatmap,
        pred_anomalous_slices=pred_anomalous_slices,
        threshold_distance=threshold_distance,
        threshold_heatmap=threshold_heatmap
    )