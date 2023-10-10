import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import tifffile
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class FlowDataset(Dataset):
    def __init__(self, root_dir, transform=transforms.Compose([transforms.ToTensor(),])):
        """
        Args:
            root_dir (string): Directory with all the images and flow files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        directories = os.listdir(root_dir)
        # Use a list comprehension to filter out .git and .gitattributes
        self.subfolders = [dir for dir in directories if dir not in [".git", ".gitattributes"]]
        self.transforms = transform
        print(True)
        

    def __len__(self):
        return sum(len(glob.glob(os.path.join(os.path.join(self.root_dir, subfolder), '*.tif')))//2 for subfolder in self.subfolders)

    def __getitem__(self, idx):
        # Find the right subfolder
        subfolder = None

        for folder in self.subfolders:
            num_images = len(glob.glob(os.path.join(os.path.join(self.root_dir, folder), '*.flo')))
            print(f"folder: {folder}, number of images {num_images}")
            if idx < num_images:
                subfolder = folder
                break
            idx -= num_images
        
        if subfolder is None:
            raise IndexError
        
        # Find the corresponding image and flow file
        image_files = sorted(glob.glob(os.path.join(os.path.join(self.root_dir,subfolder), '*.tif')))
        flow_files = sorted(glob.glob(os.path.join(os.path.join(self.root_dir,subfolder), '*.flo')))

        image_1_path = image_files[2*idx].replace("\\", "/")
        image_2_path = image_files[2*idx+1].replace("\\", "/")

        flow_path = flow_files[idx]

        image1 = tifffile.imread(image_1_path)
        image2 = tifffile.imread(image_2_path)

        flow = self.load_flo_file(flow_path)
        flow = flow.permute(2, 0, 1)
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        # # Stack images along the channel dimension
        images = torch.cat((image1, image2), dim=0)
        # images = images.permute(1, 2, 0)
        return images, flow

    def load_flo_file(self, filename):
        with open(filename, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
            else:
                w = np.fromfile(f, np.int32, count=1)[0]
                h = np.fromfile(f, np.int32, count=1)[0]
                #print('Reading %d x %d flo file' % (w, h))
                data = np.fromfile(f, np.float32, count=2*w*h)
                # Reshape data into 3D array (columns, rows, bands)
                data2D = np.resize(data, (h, w, 2))
                return torch.from_numpy(data2D)
            
