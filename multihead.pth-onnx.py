import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import os

# === 1. Определение Multi-Head модели ===
class MobileNetV3_MultiHead(nn.Module):
    def __init__(self, num_classes1=5, num_classes2=5):
        super().__init__()
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # Определение размерности входа
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.flatten(self.pool(self.features(dummy)))
            in_features = out.shape[1]

        self.head1 = nn.Linear(in_features, num_classes1)
        self.head2 = nn.Linear(in_features, num_classes2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head1(x), self.head2(x)

# === 2. Загрузка модели и весов ===
model = MobileNetV3_MultiHead()
model.load_state_dict(torch.load("models/model N4/multihead_best.pth", map_location="cpu"))
model.eval()

# === 3. Подготовка фиктивного входа с batch_size > 1 (например, 4) ===
dummy_input = torch.randn(4, 3, 224, 224)

# === 4. Экспорт с поддержкой динамического batch size ===
output_path = "multihead.onnx"
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=["input"],
    output_names=["brand", "color"],
    opset_version=11,
    dynamic_axes={
        "input": {0: "batch_size"},
        "brand": {0: "batch_size"},
        "color": {0: "batch_size"}
    }
)

