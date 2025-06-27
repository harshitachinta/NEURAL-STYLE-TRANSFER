# neural_style_transfer.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import copy
from torchvision.utils import save_image
import os

# -------------------------------
# Configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 256

# -------------------------------
# Image Loading and Preprocessing
# -------------------------------
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

def image_loader(image_path):
    image = Image.open(image_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# -------------------------------
# Content and Style Loss
# -------------------------------
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = 0.0

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = 0.0

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# -------------------------------
# Normalization Layer
# -------------------------------
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# -------------------------------
# Build the Model
# -------------------------------
def get_style_model_and_losses(cnn, mean, std, style_img, content_img):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(mean, std).to(device)
    content_layers = ['conv4_2']
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv{i}_1'
        elif isinstance(layer, nn.ReLU):
            name = f'relu{i}_1'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
            break
    model = model[:j+1]

    return model, style_losses, content_losses

# -------------------------------
# Style Transfer Execution
# -------------------------------
def run_style_transfer(cnn, mean, std, content_img, style_img, input_img,
                       num_steps=300, style_weight=1e6, content_weight=1):
    print("🔄 Starting Style Transfer...")
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, mean, std, style_img, content_img)

    optimizer = optim.LBFGS([input_img.requires_grad_()])
    run = [0]

    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_score * style_weight + content_score * content_weight
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss: {style_score:.4f}, Content Loss: {content_score:.4f}")
            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    content_img_path = "content.jpg"
    style_img_path = "style.jpg"

    if not (os.path.exists(content_img_path) and os.path.exists(style_img_path)):
        print("❗ Please make sure both 'content.jpg' and 'style.jpg' exist in the current folder.")
        exit()

    content_img = image_loader(content_img_path)
    style_img = image_loader(style_img_path)
    input_img = content_img.clone()

    assert content_img.size() == style_img.size(), "❗ Content and style images must be the same size."

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)

    imshow(output, title='🎨 Stylized Output')
    save_image(output, "stylized_output.jpg")
    print("✅ Stylized image saved as 'stylized_output.jpg'")
