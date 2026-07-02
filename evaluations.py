
import torch
from monai.metrics import DiceMetric
from monai.metrics import HausdorffDistanceMetric

def get_dice(label, prediction):
  dice_scores = []
  smooth = 1e-5  # to avoid division by zero

  for cls in [0, 1]:
        pred_cls = (prediction == cls).float()
        label_cls = (label == cls).float()

        intersection = (pred_cls * label_cls).sum()
        union = pred_cls.sum() + label_cls.sum()

        dice = (2.0 * intersection) / (union + smooth)
        dice_scores.append(dice)

  return sum(dice_scores) / len(dice_scores)



def get_total_dice(label, prediction):
    dice_metric = DiceMetric(
        include_background=True,
        reduction="mean",
        get_not_nans=False
    )

    dice_metric(y_pred=prediction, y=label)
    mean_dice_val = dice_metric.aggregate().item()
    dice_metric.reset()

    return mean_dice_val



def get_iou_score(prediction, label):
    smooth = 1e-5
    intersection = ((prediction == 1) & (label == 1)).sum().float()
    truth = (label == 1).sum().float()

    return (intersection + smooth) / (truth + smooth)



def get_total_iou_score(prediction, label):
    """
    Calculates the mean IoU for both classes (0 and 1) in a 3D image.

    Args:
        prediction (torch.Tensor): Predicted 3D tensor (binary: 0 or 1).
        label (torch.Tensor): Ground truth 3D tensor (binary: 0 or 1).

    Returns:
        float: Mean IoU across classes 0 and 1.
    """
    smooth = 1e-5
    ious = []

    for cls in [0, 1]:
        intersection = ((prediction == cls) & (label == cls)).sum().float()
        union = ((prediction == cls) | (label == cls)).sum().float()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return torch.mean(torch.stack(ious)).item()


def compute_hausdorff_distance(
    pred_raw,
    pred_post,
    label,
    include_background=True,
    percentile=95
):
    """
    Computes Hausdorff Distance (HD95) for:
    1. Raw model prediction
    2. Post-processed prediction

    Args:
        pred_raw (torch.Tensor): Raw model output [D,H,W] or [B,D,H,W]
        pred_post (torch.Tensor): Post-processed output
        label (torch.Tensor): Ground truth
    """

    device = pred_raw.device

    hd_metric = HausdorffDistanceMetric(
        include_background=include_background,
        percentile=percentile,
        reduction="mean"
    )

    # -----------------------------
    # helper: add batch + channel dims
    # -----------------------------
    def prep(x):
        if x.dim() == 3:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 4:
            x = x.unsqueeze(1)
        return x.to(device)

    pred_raw = prep(pred_raw)
    pred_post = prep(pred_post)
    label = prep(label)

    # -----------------------------
    # RAW prediction HD
    # -----------------------------
    hd_metric(y_pred=pred_raw, y=label)
    hd_raw = hd_metric.aggregate().item()
    hd_metric.reset()

    # -----------------------------
    # POST processed HD
    # -----------------------------
    hd_metric(y_pred=pred_post, y=label)
    hd_post = hd_metric.aggregate().item()
    hd_metric.reset()

    return hd_raw, hd_post