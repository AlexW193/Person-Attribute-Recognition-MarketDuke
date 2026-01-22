import os
import json
import torch
import argparse
from PIL import Image
from torchvision import transforms as T
from net import get_model
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda device name:", torch.cuda.get_device_name(0))

######################################################################
# Settings
# ---------
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}
num_cls_dict = { 'market':30, 'duke':23 }
num_ids_dict = { 'market':751, 'duke':702 }

transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


######################################################################
# Argument
# ---------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='market', type=str, help='dataset')
parser.add_argument('--backbone', default='resnet50', type=str, help='model')
parser.add_argument('--use-id', action='store_true', help='use identity loss')
parser.add_argument('input', help='Image, directory, or txt file')
parser.add_argument('--batch-size', type=int, default=256)
args = parser.parse_args()

assert args.dataset in ['market', 'duke']
assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

model_name = '{}_nfc_id'.format(args.backbone) if args.use_id else '{}_nfc'.format(args.backbone)
num_label, num_id = num_cls_dict[args.dataset], num_ids_dict[args.dataset]


######################################################################
# Resolve text file or image dir
# ---------
from pathlib import Path

def resolve_image_paths(input_arg):
    p = Path(input_arg)

    if p.is_dir():
        return sorted([
            x for x in p.iterdir()
            if x.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        ])

    if p.is_file() and p.suffix == '.txt':
        with open(p) as f:
            return [Path(line.strip()) for line in f if line.strip()]

    if p.is_file():
        return [p]

    raise ValueError(f"Invalid input: {input_arg}")

img_paths = resolve_image_paths(args.input)


######################################################################
# Image Dataset
# ---------

from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, str(path)


dataset = ImageDataset(img_paths, transforms)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
)

######################################################################
# Model and Data
# ---------
def load_network(network):
    ckpt = (
    PROJECT_ROOT
    / "checkpoints"
    / args.dataset          # ← THIS was missing
    / model_name
    / "net_last.pth"
    )


    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    network.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"[MODEL] Loaded {ckpt}")
    return network




model = get_model(model_name, num_label, use_id=args.use_id, num_id=num_id)
model = load_network(model)
model = model.to(device)
model.eval()



######################################################################
# Inference
# ---------
class predict_decoder(object):

    def __init__(self, dataset):
        doc_dir = PROJECT_ROOT / "doc"

        with open(doc_dir / "label.json", "r") as f:
            self.label_list = json.load(f)[dataset]

        with open(doc_dir / "attribute.json", "r") as f:
            self.attribute_dict = json.load(f)[dataset]

        self.num_label = len(self.label_list)

    def decode_to_dict(self, pred):
        pred = pred.cpu()
        result = {}

        for idx in range(self.num_label):
            name, choice = self.attribute_dict[self.label_list[idx]]
            if choice[pred[idx]]:
                result[name] = choice[pred[idx]]

        return result


Dec = predict_decoder(args.dataset)

model.eval()
all_results = []

with torch.no_grad():
    for imgs, paths in loader:
        imgs = imgs.to(device, non_blocking=True)

        if not args.use_id:
            out = model(imgs)
        else:
            out, _ = model(imgs)

        pred = out > 0.5  # boolean tensor

        for p, path in zip(pred, paths):
            attrs = Dec.decode_to_dict(p)   # see below
            all_results.append({
                "image": path,
                "attributes": attrs
            })


with open("results.json", "w") as f:
    json.dump(all_results, f, indent=2)
    