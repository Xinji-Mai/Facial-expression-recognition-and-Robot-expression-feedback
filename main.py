from flask import Flask, request, Response
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
import json
from PIL import Image
import os

# 设置设备
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5"
USE_CUDA = torch.cuda.is_available()
device_ids_parallel = [3]
device = torch.device("cuda:{}".format(device_ids_parallel[0]) if USE_CUDA else "cpu")

app = Flask(__name__)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# 加载训练好的模型
model = VGG('VGG19')
checkpoint = torch.load('your_model',map_location = device)
model.load_state_dict(checkpoint['net'])
model.to(device)
model.eval()

# 确保预处理步骤与训练时一致
cut_size = 44
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image file", 400

    file = request.files['image']
    img = Image.open(file.stream)

    # 预处理图像
    gray = img.convert('L')
    gray = gray.resize((48, 48))
    img = np.stack([gray] * 3, axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.to(device)
    outputs = model(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
    score = F.softmax(outputs_avg, dim=0)
    _, predicted = torch.max(outputs_avg.data, 0)

    pred_label = class_names[predicted.item()]

    return Response(json.dumps(pred_label), mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

