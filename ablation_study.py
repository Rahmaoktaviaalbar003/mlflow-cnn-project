
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "CNN-FashionMNIST"

SEED = 42
EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.0062189838789497

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Dataset
transform = transforms.ToTensor()

dataset = datasets.FashionMNIST(
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
    dataset,
    [54000, 6000],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    generator=torch.Generator().manual_seed(SEED)
)

val_loader = DataLoader(
    val_dataset, batch_size=128, shuffle=False
)

# Tiga arsitektur untuk dibandingkan
class CNN(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant

        if variant == "baseline":
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            input_size = 64 * 7 * 7

        elif variant == "one_conv":
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            input_size = 32 * 14 * 14

        elif variant == "no_first_pool":
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            input_size = 64 * 14 * 14

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


variants = [
    ("baseline", "Baseline-2Conv-2Pool"),
    ("one_conv", "Ablation-1Conv"),
    ("no_first_pool", "Ablation-NoFirstPool")
]

results = {}

with mlflow.start_run(run_name="Ablation-Study"):

    mlflow.log_params({
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED
    })

    for variant, run_name in variants:
        torch.manual_seed(SEED)

        model = CNN(variant).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE
        )

        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_params({
                "architecture_variant": variant,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "seed": SEED
            })

            for epoch in range(EPOCHS):
                model.train()

                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    loss = criterion(model(images), labels)
                    loss.backward()
                    optimizer.step()

            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    correct += (outputs.argmax(1) == labels).sum().item()
                    total += labels.size(0)

            val_accuracy = correct / total
            results[variant] = val_accuracy

            mlflow.log_metric("val_accuracy", val_accuracy)

            print(f"{run_name}: validation accuracy = {val_accuracy:.4f}")

    mlflow.log_dict(results, "ablation_results.json")

print("\nAblation study selesai.")
print("Hasil dicatat di MLflow.")
