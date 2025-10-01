import matplotlib.pyplot as plt
import torchvision
from data_loader import SmokerDataModule

def show_batch(loader, class_names, n=8):
    xb, yb = next(iter(loader))  # one batch
    grid = torchvision.utils.make_grid(xb[:n], nrow=4, normalize=True, pad_value=1)
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title(" | ".join([class_names[label] for label in yb[:n]]))
    plt.show()

if __name__ == "__main__":
    dm = SmokerDataModule(data_dir="../data", batch_size=16, num_workers=0)
    dm.setup()

    print("Class to index mapping:", dm.train_dataset.class_to_idx)

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    print("Train size:", len(dm.train_dataset))
    print("Val size:", len(dm.val_dataset))
    print("Test size:", len(dm.test_dataset))

    # Show a batch of training images
    show_batch(train_loader, dm.train_dataset.classes, n=8)

    print(torchvision.__version__)