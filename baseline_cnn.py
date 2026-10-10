
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 1. Konfigurasi MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("CNN-FashionMNIST")

torch.manual_seed(42)

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# 2. Membaca hash dataset
hash_file = Path("dataset_hash.json")

if not hash_file.exists():
    raise FileNotFoundError(
        "dataset_hash.json tidak ditemukan. "
        "Jalankan python dataset_hash.py terlebih dahulu."
    )

with hash_file.open("r", encoding="utf-8") as f:
    dataset_info = json.load(f)

dataset_hash = dataset_info["dataset_sha256"]

print("Dataset:", dataset_info["dataset"])
print("Dataset SHA-256:", dataset_hash)


# 3. Dataset Fashion-MNIST
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# 4. Arsitektur CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# 5. Training dan pencatatan MLflow
with mlflow.start_run(run_name="CNN-Baseline"):

    mlflow.log_params({
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "optimizer": "Adam",
        "architecture": "2-layer CNN",
        "dataset": "Fashion-MNIST",
        "seed": 42
    })

    # Catat identitas dataset pada run
    mlflow.set_tag("dataset_sha256", dataset_hash)
    mlflow.set_tag("dataset_name", dataset_info["dataset"])
    mlflow.set_tag(
        "dataset_file_count",
        str(dataset_info["file_count"])
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = correct / total

        # Evaluasi pada test dataset
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                test_correct += (
                    outputs.argmax(1) == labels
                ).sum().item()
                test_total += labels.size(0)

        test_accuracy = test_correct / test_total

        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        }, step=epoch + 1)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Test Acc: {test_accuracy:.4f}"
        )

    # Simpan model ke MLflow
    example_input = torch.zeros(1, 1, 28, 28)
    model_cpu = model.to("cpu")

    mlflow.pytorch.log_model(
        model_cpu,
        name="cnn_model",
        input_example=example_input,
        serialization_format="pickle"
    )

    print("Baseline selesai.")
    print("Metrik, model, dan hash dataset tercatat di MLflow.")
