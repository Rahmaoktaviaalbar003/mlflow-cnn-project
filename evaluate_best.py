
import json
import statistics

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Konfigurasi
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "CNN-FashionMNIST"

BEST_LR = 0.0062189838789497
BATCH_SIZE = 64
FILTERS = 32
EPOCHS = 2
SEEDS = [42, 123, 2026]

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Dataset
transform = transforms.ToTensor()

full_train = datasets.FashionMNIST(
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

train_dataset, val_dataset = random_split(
    full_train,
    [54000, 6000],
    generator=torch.Generator().manual_seed(42)
)

# Arsitektur CNN
class CNN(nn.Module):
    def __init__(self, filters):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(filters, filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters * 2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

test_accuracies = []

# Parent run untuk evaluasi tiga seed
with mlflow.start_run(run_name="Best-Model-3-Seeds"):

    mlflow.log_params({
        "learning_rate": BEST_LR,
        "batch_size": BATCH_SIZE,
        "filters": FILTERS,
        "epochs": EPOCHS,
        "seeds": json.dumps(SEEDS),
        "evaluation": "3-seed test evaluation",
        "dataset": "Fashion-MNIST"
    })

    for seed in SEEDS:
        torch.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed)
        )

        model = CNN(FILTERS).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=BEST_LR
        )

        with mlflow.start_run(
            run_name=f"Best-Seed-{seed}",
            nested=True
        ):
            mlflow.log_params({
                "seed": seed,
                "learning_rate": BEST_LR,
                "batch_size": BATCH_SIZE,
                "filters": FILTERS,
                "epochs": EPOCHS
            })

            # Training
            for epoch in range(EPOCHS):
                model.train()
                total_loss = 0.0
                total_samples = 0

                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * images.size(0)
                    total_samples += images.size(0)

                train_loss = total_loss / total_samples
                mlflow.log_metric(
                    "train_loss",
                    train_loss,
                    step=epoch + 1
                )

            # Evaluasi test setelah training selesai
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    correct += (
                        outputs.argmax(1) == labels
                    ).sum().item()
                    total += labels.size(0)

            test_accuracy = correct / total
            test_accuracies.append(test_accuracy)

            mlflow.log_metric("test_accuracy", test_accuracy)

            print(
                f"Seed {seed} | "
                f"Test accuracy: {test_accuracy:.4f}"
            )

    mean_accuracy = statistics.mean(test_accuracies)
    std_accuracy = statistics.stdev(test_accuracies)

    mlflow.log_metrics({
        "mean_test_accuracy": mean_accuracy,
        "std_test_accuracy": std_accuracy
    })

    mlflow.log_dict(
        {
            "seeds": SEEDS,
            "test_accuracies": test_accuracies,
            "mean_test_accuracy": mean_accuracy,
            "std_test_accuracy": std_accuracy
        },
        "three_seed_results.json"
    )

    print("\n=== HASIL 3 SEED ===")
    print(f"Mean test accuracy: {mean_accuracy:.4f}")
    print(f"Std test accuracy:  {std_accuracy:.4f}")

print("\nEvaluasi 3 seed selesai.")
