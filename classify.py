import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from PIL import Image

MODEL_PATH = "models/plant_resnet18_ft.pth"
CONF_THRESHOLD = 0.70   # tune after testing on outside photos


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        idn = x if self.downsample is None else self.downsample(x)
        return F.relu(out + idn)


class ResNet18(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.fc = nn.Linear(512, n)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


_ckpt = torch.load(MODEL_PATH, map_location="cpu")
CLASSES = _ckpt["classes"]
_model = ResNet18(len(CLASSES))
_model.load_state_dict(_ckpt["model"])
_model.eval()

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _prep(img):
    img = img.convert("RGB").resize((128, 128), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return ((x - _MEAN) / _STD).unsqueeze(0)


def _clean(name):
    return name.replace("_", " ").strip()


def crop_supported(crop):
    crops = [_clean(c.split("___")[0]).lower() for c in CLASSES]
    return any(crop.lower().strip() in c for c in crops)


def classify_leaf(image, top_k=3):
    with torch.no_grad():
        probs = torch.softmax(_model(_prep(image)), 1)[0]
    top = probs.topk(top_k)
    out = []
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        crop, _, disease = CLASSES[i].partition("___")
        out.append({"crop": _clean(crop), "disease": _clean(disease),
                    "confidence": round(p, 2)})
    return out


def describe_prediction(results):
    top = results[0]
    if top["confidence"] < CONF_THRESHOLD:
        return ("The photo classifier is NOT confident about this image "
                f"(best guess {top['crop']} - {top['disease']}, {top['confidence']:.0%}). "
                "Do not rely on it.")
    others = "; ".join(f"{r['crop']} - {r['disease']} ({r['confidence']:.0%})"
                       for r in results[1:])
    return (f"The photo classifier suggests {top['crop']} - {top['disease']} "
            f"({top['confidence']:.0%}). Other possibilities: {others}. "
            "Treat this only as a hint, not a confirmed diagnosis.")