import torch
import sys
import os
from monai.losses import DiceCELoss
from .UNETR import UNETR

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)
from config import (
    IMG_SIZE,
    NUM_CLASSES,
    LR,
    DEVICE,
    MODEL_WEIGHTS,
)

def build_model():
    """
    Create UNETR model
    """
    model = UNETR(
        in_channels=1,
        out_channels=NUM_CLASSES,
        img_size=IMG_SIZE,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
    )

    return model


def load_pretrained_weights(model, weight_path):
    """
    Load pretrained checkpoint
    """
    print(f"Loading weights from: {weight_path}")

    state_dict = torch.load(weight_path, map_location="cpu")

    if "out.conv.conv.weight" in state_dict:
        state_dict["out.conv.conv.weight"] = state_dict[
            "out.conv.conv.weight"
        ][:NUM_CLASSES]

    if "out.conv.conv.bias" in state_dict:
        state_dict["out.conv.conv.bias"] = state_dict[
            "out.conv.conv.bias"
        ][:NUM_CLASSES]

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False
    )

    print("Weights loaded successfully")

    if missing_keys:
        print("\nMissing Keys:")
        print(missing_keys)

    if unexpected_keys:
        print("\nUnexpected Keys:")
        print(unexpected_keys)

    return model


def create_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )


def create_loss():
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        squared_pred=True
    )


def main():
    device = torch.device(
        DEVICE if torch.cuda.is_available() else "cpu"
    )

    model = build_model()
    model = load_pretrained_weights(model, MODEL_WEIGHTS)

    model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    main()