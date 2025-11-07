# fnst.py
import torch
from torchvision import transforms
from PIL import Image
import os
from .transformer_net import TransformerNet

# Load the model once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerNet()

weights_path = os.path.join("saved_models", "mosaic.pth")
state_dict = torch.load(weights_path)

for k in list(state_dict.keys()):
    if k.endswith(('running_mean', 'running_var')):
        del state_dict[k]

model.load_state_dict(state_dict, strict=False)
model.to(device).eval()

# Image loader
def load_image(img_path, size=512):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Image saver
def save_image(tensor, filename):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image / 255.0)
    image.save(filename)

# 🔧 Main style transfer function
def stylize_image(input_path, output_path):
    content_img = load_image(input_path)
    with torch.no_grad():
        output = model(content_img)
    save_image(output, output_path)
    return output_path
