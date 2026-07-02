from .IntermediateLayerExtractor import IntermediateLayerExtractor
import numpy as np
import torch
import torch.nn.functional as F

def get_built_model(model, layer_name):
    gradcam_model = IntermediateLayerExtractor(
        model,
        layer_name
    )

    gradcam_model.eval()

    return gradcam_model


def chunk_images_from_depth(
    image_array,
    patch_slice_number
):
    divide_integer = image_array.shape[0] // patch_slice_number
    reminder = image_array.shape[0] % patch_slice_number

    patch_list = []

    patch_start = 0
    patch_end = patch_slice_number

    for _ in range(divide_integer):
        ct_volume = image_array[
            patch_start:patch_end,
            :,
            :
        ]

        patch_list.append(ct_volume)

        patch_start += patch_slice_number
        patch_end += patch_slice_number

    last_patch_end_index = patch_slice_number - reminder

    if reminder > 0:
        start_idx = (
            image_array.shape[0]
            - patch_slice_number
        )

        last_patch = image_array[start_idx:, :, :]
        patch_list.append(last_patch)

    return patch_list, last_patch_end_index


def get_gradcam_result(
    gradcam_model,
    image,
    class_index
):
    device = next(
        gradcam_model.parameters()
    ).device

    image = image.to(device)
    image.requires_grad_(True)

    conv_outputs, predictions = gradcam_model(image)

    score = predictions[:, class_index].mean()

    gradcam_model.zero_grad()

    conv_outputs.retain_grad()

    score.backward(retain_graph=True)

    feature_maps = conv_outputs.detach().cpu()
    feature_maps_gradients = (
        conv_outputs.grad.detach().cpu()
    )

    gate_feature_maps_gradients = (
        feature_maps_gradients > 0
    )

    gate_feature_maps = (
        feature_maps > 0
    )

    guided_grads = (
        gate_feature_maps_gradients.float()
        * gate_feature_maps.float()
        * feature_maps_gradients
    )

    weights = torch.mean(
        guided_grads,
        dim=(2, 3, 4)
    )

    cam = torch.zeros(
        feature_maps.shape[2:],
        dtype=torch.float32
    )

    for i, w in enumerate(weights[0]):
        cam += (
            w
            * feature_maps[0, i]
        )

    cam = cam.unsqueeze(0).unsqueeze(0)

    cam_resized = F.interpolate(
        cam,
        size=(
            image.shape[2],
            image.shape[3],
            image.shape[4]
        ),
        mode="trilinear",
        align_corners=False
    )

    cam_resized = (
        cam_resized
        .squeeze(0)
        .squeeze(0)
        .numpy()
    )

    cam_resized = np.maximum(
        cam_resized,
        0
    )

    heatmap = (
        cam_resized - cam_resized.min()
    ) / (
        cam_resized.max()
        - cam_resized.min()
        + 1e-8
    )

    return heatmap


def generate_gradcam(
    image,
    model,
    layer_name,
    class_index
):
    gradcam_model = get_built_model(
        model,
        layer_name
    )

    heatmap = get_gradcam_result(
        gradcam_model,
        image,
        class_index
    )

    return heatmap


def main(
    image,
    model,
    layer_name="decoder3",
    class_index=1
):
    """
    Main entry point for Grad-CAM generation.

    Parameters
    ----------
    image : torch.Tensor
    model : torch.nn.Module
    layer_name : str
    class_index : int

    Returns
    -------
    numpy.ndarray
        3D Grad-CAM heatmap
    """

    return generate_gradcam(
        image=image,
        model=model,
        layer_name=layer_name,
        class_index=class_index
    )