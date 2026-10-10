
import torch
import mlflow
import mlflow.pytorch

from torchvision import datasets, transforms

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_URI = "models:/FashionMNIST-CNN@champion"

mlflow.set_tracking_uri(TRACKING_URI)

# Muat model melalui alias champion
model = mlflow.pytorch.load_model(MODEL_URI)
model.eval()

# Ambil satu gambar dari test dataset
dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

image, true_label = dataset[0]
image = image.unsqueeze(0)

# Prediksi
with torch.no_grad():
    output = model(image)
    predicted_label = output.argmax(dim=1).item()

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

print("Model URI:", MODEL_URI)
print("True label:", class_names[true_label])
print("Predicted label:", class_names[predicted_label])
print("Prediction correct:", predicted_label == true_label)
