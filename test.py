import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import pickle
from models import *

#define dataset with images from the test folder
class Test_dataset(Dataset):
    def __init__(self, images_dir='test_images'):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.max_name_length = max([len(f) for f in self.ids])
    
    def __getitem__(self, i):
        image = Image.open(self.images_fps[i])
        image = image.convert('RGB')
        name = self.ids[i]
        image = transforms.Resize((32,32), interpolation=2)(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))(image)
        return image, name
        
    def __len__(self):
        return len(self.ids)

    def max_name_length(self):
        return self.max_name_length

#dummy datasets needed to download CIFAR100 if this script is run before train.py to make sure the cifar labels are available
if not os.path.isdir("cifar100"):
    dummy_train = datasets.CIFAR100('cifar100', train=True, download=True)
    dummy_val = datasets.CIFAR100('cifar100', train=False, download=True)

#read labels
with open("cifar100/cifar-100-python/meta", 'rb') as f:
    labels_dict = pickle.load(f)
labels = labels_dict["fine_label_names"]

test_ds = Test_dataset()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#test all models in the folder "trained models"
print("Testing on", device)
for model_name in os.listdir("./trained_models"):
    #get width multiplier from the model name
    wm = float(model_name.split('_')[3])
    #get model size (large or small)
    if model_name.split('_')[2] == "small":
        model = Mobilenet_v3_small(wm)
    else:
        model = Mobilenet_v3_large(wm)

    model.load_state_dict(torch.load(os.path.join("./trained_models", model_name), map_location=device))
    model = model.to(device)
    model.eval()
    print("Testing", model.name(), model_name)
    with torch.no_grad():
        for img, img_name in test_ds:
            img = img.unsqueeze(0)
            img = img.to(device)
            outputs = model(img)
            outputs = nn.Softmax(dim=1)(outputs)
            conf, pred = torch.max(outputs, dim=1)
            print('{:{name_width}s} {:15s}  {:3.2f}'.format(img_name, labels[pred.item()], conf.item(), name_width=test_ds.max_name_length))
    print("")
