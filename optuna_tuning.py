import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import optuna

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# 1. Konfigurasi
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "CNN-FashionMNIST"
N_TRIALS = 20
EPOCHS_PER_TRIAL = 1
SEED = 42

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Device: {device}")


# 2. Membaca hash dataset
def log_dataset_hash():
    hash_file = Path("dataset_hash.json")

    if not hash_file.exists():
        print(
            "PERINGATAN: dataset_hash.json tidak ditemukan. "
            "Hash dataset tidak dicatat pada run ini."
        )
        return

    with hash_file.open("r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    mlflow.set_tag(
        "dataset_sha256",
        dataset_info["dataset_sha256"]
    )
    mlflow.set_tag(
        "dataset_name",
        dataset_info["dataset"]
    )
    mlflow.set_tag(
        "dataset_file_count",
        str(dataset_info["file_count"])
    )


# 3. Dataset dan validation split
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
    generator=torch.Generator().manual_seed(SEED)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)


# 4. CNN dengan jumlah filter yang bisa diatur
class CNN(nn.Module):
    def __init__(self, filters):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1, filters,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                filters, filters * 2,
                kernel_size=3,
                padding=1
            ),
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


# 5. Fungsi objective Optuna
def objective(trial):
    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        1e-2,
        log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128]
    )

    filters = trial.suggest_categorical(
        "filters",
        [16, 32]
    )

    # Seed yang sama agar perbandingan lebih konsisten
    torch.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False
    )

    model = CNN(filters).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    with mlflow.start_run(
        run_name=f"Optuna-Trial-{trial.number:02d}",
        nested=True
    ):
        # Catat hash dataset untuk setiap trial
        log_dataset_hash()

        mlflow.log_params({
            "trial_number": trial.number,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "filters": filters,
            "epochs": EPOCHS_PER_TRIAL,
            "optimizer": "Adam",
            "dataset": "Fashion-MNIST",
            "seed": SEED
        })

        # Training
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

        # Evaluasi validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                correct += (
                    outputs.argmax(1) == labels
                ).sum().item()

                total += labels.size(0)

        val_accuracy = correct / total

        # Catat metrik
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_accuracy": val_accuracy
        })

        trial.set_user_attr("train_loss", train_loss)

        print(
            f"Trial {trial.number + 1}/{N_TRIALS} | "
            f"LR: {learning_rate:.5f} | "
            f"Batch: {batch_size} | "
            f"Filters: {filters} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

    return val_accuracy


# 6. Jalankan 20 trials sebagai nested runs
with mlflow.start_run(
    run_name="Optuna-Tuning-20-Trials"
):
    # Catat hash dataset pada run induk juga
    log_dataset_hash()

    mlflow.log_params({
        "study": "CNN-Hyperparameter-Tuning",
        "n_trials": N_TRIALS,
        "epochs_per_trial": EPOCHS_PER_TRIAL,
        "validation_split": 0.1,
        "seed": SEED
    })

    study = optuna.create_study(
        direction="maximize",
        study_name="FashionMNIST-CNN"
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS
    )

    best_params = study.best_params
    best_accuracy = study.best_value

    mlflow.log_metric(
        "best_val_accuracy",
        best_accuracy
    )

    mlflow.log_dict(
        best_params,
        "best_params.json"
    )

    print("\n=== HASIL OPTUNA ===")
    print(
        f"Best validation accuracy: {best_accuracy:.4f}"
    )
    print("Best parameters:")
    print(json.dumps(best_params, indent=4))


print("\nOptuna tuning selesai.")
print("Hasil setiap trial tercatat di MLflow.")