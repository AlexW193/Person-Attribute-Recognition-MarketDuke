import os
import json
import torch
import argparse
from PIL import Image
from torchvision import transforms as T
from net import get_model

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
parser.add_argument('image_path', nargs='+', help='Path(s) to test image(s)')
parser.add_argument('--dataset', default='market', type=str, help='dataset')
parser.add_argument('--backbone', default='resnet50', type=str, help='model')
parser.add_argument('--use-id', action='store_true', help='use identity loss')
args = parser.parse_args()

assert args.dataset in ['market', 'duke']
assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

model_name = '{}_nfc_id'.format(args.backbone) if args.use_id else '{}_nfc'.format(args.backbone)
num_label, num_id = num_cls_dict[args.dataset], num_ids_dict[args.dataset]


######################################################################
# Model and Data
# ---------
def load_network(network):
    ckpt = (
        PROJECT_ROOT
        / "checkpoints"
        / attribute_detector
        / model_name
        / "net_last.pth"
    )

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    network.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"[MODEL] Loaded {ckpt}")
    return network

def load_images(paths):
    imgs = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        img = transforms(img)
        imgs.append(img)
    return torch.stack(imgs, dim=0)  # [B, C, H, W]

model = get_model(model_name, num_label, use_id=args.use_id, num_id=num_id)
model = load_network(model)
model = model.to(device)
model.eval()

src = load_images(args.image_path).to(device)


######################################################################
# Inference
# ---------
class predict_decoder(object):

    def __init__(self, dataset):
        with open('./doc/label.json', 'r') as f:
            self.label_list = json.load(f)[dataset]
        with open('./doc/attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self, pred):
        # pred: [B, num_label]
        for b in range(pred.size(0)):
            print(f'\nImage {b}:')
            for idx in range(self.num_label):
                name, chooce = self.attribute_dict[self.label_list[idx]]
                if chooce[pred[b, idx]]:
                    print(f'{name}: {chooce[pred[b, idx]]}')



with torch.no_grad():
    if not args.use_id:
        out = model.forward(src)
    else:
        out, _ = model.forward(src)

pred = torch.gt(out, torch.ones_like(out)/2 )  # threshold=0.5

Dec = predict_decoder(args.dataset)
Dec.decode(pred)

