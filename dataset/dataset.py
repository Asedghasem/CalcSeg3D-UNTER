import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from .Dataset3D import Dataset3D
from torch.utils.data import DataLoader

def main():
    print("====================================")
    print("🚀 Loading config and datasets...")
    print("====================================")

    print(f"Train dir: {config.TRAIN_DIR}")
    print(f"Val dir: {config.VAL_DIR}")
    print(f"Test dir: {config.TEST_DIR}")

    train_dataset = Dataset3D(config.TRAIN_DIR)
    val_dataset = Dataset3D(config.VAL_DIR)
    test_dataset = Dataset3D(config.TEST_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False
    )

    print("====================================")
    print("📊 Dataset sizes")
    print("====================================")
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    print("====================================")
    print("🔄 Testing one batch from train_loader")
    print("====================================")

    for images, labels, paths in train_loader:
        print("Batch image shape:", images.shape)
        print("Batch label shape:", labels.shape)
        print("Sample path:", paths[0])
        break

    print("====================================")
    print("✅ Done")
    print("====================================")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    main()