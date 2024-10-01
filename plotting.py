import random
import torch
import torchvision
import matplotlib.pyplot as plt

from typing import Dict, List

from utils import make_predictions


def plot_loss_curve(results: Dict[str, List[float]]):
    """Plots training curves using results dictionary"""
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    acc = results["train_acc"]
    test_acc = results["test_acc"]

    epochs = range(len(results["train_acc"]))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, test_loss, label="Test Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, test_acc, label="Test Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


def make_and_plot_predictions(model: torch.nn.Module,
                              data: torchvision.datasets,
                              class_names: Dict[int, str]):
    test_labels = []
    test_samples = []

    for sample, label in random.sample(list(data), k=9):
        test_labels.append(label)
        test_samples.append(sample)

    pred_probs = make_predictions(model=model, data=test_samples)
    pred_classes = pred_probs.argmax(dim=1)

    rows, cols = 3, 3
    plt.figure(figsize=(9, 9))

    for i, sample in enumerate(test_samples):
        plt.axis(False)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(sample.permute((1, 2, 0)))
        pred_label = class_names[pred_classes[i]]
        truth_label = class_names[test_labels[i]]
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"

        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g")
        else:
            plt.title(title_text, fontsize=10, c="r")

    plt.show()
