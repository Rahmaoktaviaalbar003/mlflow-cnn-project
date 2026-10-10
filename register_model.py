
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from mlflow import MlflowClient
from mlflow.models import infer_signature
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "CNN-FashionMNIST"
MODEL_NAME = "FashionMNIST-CNN"

SEED = 42
EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.0062189838789497
FILTERS = 32

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

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
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

# Model CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, FILTERS, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(FILTERS, FILTERS * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FILTERS * 2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

with mlflow.start_run(run_name="Final-Registered-CNN") as run:
    mlflow.log_params({
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "filters": FILTERS,
        "epochs": EPOCHS,
        "seed": SEED,
        "dataset": "Fashion-MNIST",
        "architecture": "2-layer CNN"
    })

    # Training
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
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
            total += labels.size(0)

        mlflow.log_metric(
            "train_loss",
            total_loss / total,
            step=epoch + 1
        )

    # Validation evaluation
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total
    mlflow.log_metric("val_accuracy", val_accuracy)

    # Final test evaluation
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_correct += (outputs.argmax(1) == labels).sum().item()
            test_total += labels.size(0)

    test_accuracy = test_correct / test_total
    mlflow.log_metric("test_accuracy", test_accuracy)

    # Log model with a tensor signature
    example_input = torch.zeros(1, 1, 28, 28)
    model_cpu = model.to("cpu")
    example_output = model_cpu(example_input).detach().numpy()

    signature = infer_signature(
        example_input.numpy(),
        example_output
    )

    mlflow.pytorch.log_model(
        model_cpu,
        name="cnn_model",
        signature=signature,
        serialization_format="pickle"
    )

    model_uri = f"runs:/{run.info.run_id}/cnn_model"
    print("Validation accuracy:", round(val_accuracy, 4))
    print("Test accuracy:", round(test_accuracy, 4))
    print("Model URI:", model_uri)

# Register model version
registered = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

client = MlflowClient(tracking_uri=TRACKING_URI)

client.set_registered_model_alias(
    MODEL_NAME,
    "champion",
    registered.version
)

print("Registered model:", MODEL_NAME)
print("Version:", registered.version)
print("Alias: champion")
print("Registrasi model selesai.")
