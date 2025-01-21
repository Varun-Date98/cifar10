import os

# Importing PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader

# Importing torchvision components
from torchvision import datasets
from torchvision.transforms import ToTensor

# Importing custom components
from model import TinyVGG
from utils import train_model
from plotting import plot_loss_curve, make_and_plot_predictions


BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=None
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        target_transform=None
    )

    class_names = train_data.classes
    print(class_names)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)

    tiny_vgg_model = TinyVGG(input_shape=3, num_classes=len(class_names)).to(DEVICE)
    model_save_path = "/".join(["./saved_models", tiny_vgg_model.__class__.__name__ + ".pth"])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=tiny_vgg_model.parameters(), lr=1e-3, weight_decay=1e-4)

    if os.path.exists(model_save_path):
        tiny_vgg_model.load_state_dict(torch.load(model_save_path))
    else:
        model_results = train_model(model=tiny_vgg_model,
                                    train_dataloader=train_dataloader,
                                    test_dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    optimizer=optimizer,
                                    epochs=15,
                                    device=DEVICE)

        plot_loss_curve(model_results)

    make_and_plot_predictions(model=tiny_vgg_model, data=test_data, class_names=class_names)


if __name__ == "__main__":
    main()
