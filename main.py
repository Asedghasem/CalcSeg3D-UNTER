from dataset.dataset import main as dataset_main
from models.train import main as train_main
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from gradcam.gradcam import main as gradcam_main
from postProcessing import main as post_processing
from evaluations import get_dice
from evaluations import get_iou_score
from evaluations import compute_hausdorff_distance
from evaluations import get_total_dice
from evaluations import get_total_iou_score
from loadData import read_test_image
import torch

post_label = AsDiscrete(to_onehot=True, num_classes=2)
post_pred = AsDiscrete(argmax=True, to_onehot=True, num_classes=2)
dice_metric = DiceMetric(
    include_background=True,
    reduction="mean",
    get_not_nans=False
)

def main():
    print("Step 1: Loading dataset...")
    train_loader, val_loader, test_loader = dataset_main()

    print("Step 2: Loading model...")
    model = train_main()

    image_index = 5
    image, label = read_test_image(image_index, test_loader)

    print("Step 3: Generating GradCam...")
    heatmap = gradcam_main(
        image=image,
        model=model,
        layer_name="decoder3",
        class_index=1
    )

    print("Heatmap shape:", heatmap.shape)

    print("Step 4: Predicting Data...")
    val_outputs, output, dice = prediction(
        image,
        label,
        model
    )

    print("Dice:", dice)
    print("Output shape:", output.shape)

    print("Step 5: Post-processing Data...")
    pred_slices_with_anomaly = output[0].any(dim=(1, 2))
    pred_anomalous_slices = pred_slices_with_anomaly.nonzero(as_tuple=True)[0].tolist()
    expanded_mask = post_processing(output[0].cpu(), heatmap, pred_anomalous_slices, 1, 0.7)
    expanded_mask = torch.from_numpy(expanded_mask)
    print(expanded_mask.shape)

    print("Everything is ready.")

    print("Evaluations Results")
    print(f"Total Expanded Dice = {get_dice(label[0][0].cpu(), expanded_mask.cpu())}")
    print(f"Total Segmentation Dice = {get_dice(label[0][0].cpu(), output[0].cpu())}")
    print(f"IOU Befor Expand : {get_iou_score(output[0].cpu(), label[0][0])}")
    print(f"IOU After Expand : {get_iou_score(expanded_mask.cpu(), label[0][0])}")
    print()

    hd_raw, hd_post = compute_hausdorff_distance(
        pred_raw=output[0].cpu(),
        pred_post=expanded_mask.cpu(),
        label=label[0].cpu()
    )

    print(f"HD95 Raw Prediction: {hd_raw}")
    print(f"HD95 Post Processed: {hd_post}")


def prediction(image, label, model, uncertainty_threshold = 0.3):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        image = image.to(device)
        label = label.to(device)
        model = model.to(device)

        val_outputs = sliding_window_inference(
            image, (96, 96, 96), 4, model
        )

        val_labels_list = decollate_batch(label)
        val_labels_convert = [post_label(x) for x in val_labels_list]

        val_outputs_list = decollate_batch(val_outputs)
        val_output_convert = [post_pred(x) for x in val_outputs_list]

        dice_metric(y_pred=val_output_convert, y=val_labels_convert)
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()

        output = torch.argmax(val_outputs, dim=1)

    return val_outputs, output, mean_dice_val


if __name__ == "__main__":
    main()