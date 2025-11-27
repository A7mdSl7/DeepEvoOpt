# from src.utils.data_loader import get_dataloaders

# train_loader, val_loader, test_loader, classes = get_dataloaders(
#     dataset_name="fashion-mnist",
#     data_dir="data/raw",
#     batch_size=32,
#     val_split=0.15,
#     test_split=0.15,
#     num_workers=0,   # لو على Windows وزيادة الأمان خليها 0 أو 2
#     resize=32,       # لو عايز تكبّر الصور عشان تستخدم CNN ب conv layers
#     seed=42
# )

# print("Classes:", classes)
# batch = next(iter(train_loader))
# images, labels = batch
# print("Batch images shape:", images.shape)  # expected: [B, C, H, W]
# print("Batch labels shape:", labels.shape)


# src/utils/check_data.py
from src.utils.data_loader import get_dataloaders
import torch


def main():
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        dataset_name="fashion-mnist",
        data_dir="data/raw",
        batch_size=64,
        val_split=0.15,
        test_split=0.15,
        num_workers=0,
        resize=28,   # 28x28 original
        seed=42
    )
    print("Classes:", classes)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    imgs, labels = next(iter(train_loader))
    print("Batch images shape:", imgs.shape)   # expected: [32, 1, 28, 28]
    print("Batch labels shape:", labels.shape)
    print("Sample labels:", labels[:10].tolist())


if __name__ == "__main__":
    main()
