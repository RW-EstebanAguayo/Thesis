import os
from PIL import Image
import torch
import cv2
from torch.utils.data import Dataset, DataLoader

class PIVDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.flow_types = os.listdir(root_dir)
        self.num_samples = self.calculate_num_samples()
    
    def calculate_num_samples(self):
        num_samples = 0
        for flow_type in self.flow_types:
            flow_dir = os.path.join(self.root_dir, flow_type)
            file_names = [f for f in os.listdir(flow_dir) if f.endswith("_img1.tif")]
            num_samples += len(file_names)
        return num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        flow_type_idx = 0
        while idx >= len(os.listdir(os.path.join(self.root_dir, self.flow_types[flow_type_idx]))):
            idx -= len(os.listdir(os.path.join(self.root_dir, self.flow_types[flow_type_idx])))
            flow_type_idx += 1
        flow_type = self.flow_types[flow_type_idx]
        flow_dir = os.path.join(self.root_dir, flow_type)

        file_names = [f for f in os.listdir(flow_dir) if f.endswith("_img1.tif")]
        instance_name = file_names[idx]

        img1_path = os.path.join(flow_dir, instance_name)
        img2_path = os.path.join(flow_dir, instance_name.replace("_img1.tif", "_img2.tif"))
        flow_path = os.path.join(flow_dir, instance_name.replace("_img1.tif", "_flow.flo"))

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        flow = load_velocity_field(flow_path)  # Implement a function to load .flo files

        return {"image_pair": (img1, img2), "velocity_field": flow}

    def load_velocity_field(flow_path):
        # Use OpenCV to read the .flo file
        flow = cv2.readOpticalFlow(flow_path)
        # Convert the loaded flow data to a NumPy array
        flow = flow[..., 0:2]  # Extract the U and V components
        return flow
    # Create an instance of the custom dataset
dataset = PIVDataset(root_dir='C:/Users/estal/OneDrive - ROCKWOOL Group/Documents/GD/Thesis/04_Code/src/data/raw/PIV_dataset/PIV-genImages/data')

# Create a data loader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("success!")

