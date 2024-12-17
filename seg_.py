import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from imageio import imread
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from glob import glob


writer = SummaryWriter(comment='Segementation')

torch.manual_seed(2023)

paths = glob("./stage1_train/*")

class DSB2018(Dataset):
    """Dataset class for the 2018 Data Science Bowl."""

    def __init__(self, paths):
        """paths: a list of paths to every image folder in the dataset"""
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # There is only one image in each images path. So we will grab the "first" thing we find with "[0]" at the end
        img_path = glob(self.paths[idx] + "/images/*")[0]
        # but there are multiple mask images in each mask path
        mask_imgs = glob(self.paths[idx] + "/masks/*")
        # the image shape is (W, H, 4), the last dimension is an 'alpha' channel that is not used
        img = imread(img_path)[:, :, 0:3]  # trim off the alpha so we get (W, H, 3)
        # Now we want this as (3, W, H), which is the normal shape for PyTorch
        img = np.moveaxis(img, -1, 0)
        # Last step for the image, re-scale it to the range [0, 1]
        img = img / 255.0

        # Every mask image is going to have a shape of (W, H) which has a value of 1 if the pixel is of a nuclei, and a value of 0 if the image is background/ a  _different_ nuclei
        masks = [imread(f) / 255.0 for f in mask_imgs]

        # Since we want to do simple segmentation, we will create one final mask that contains _all_ nuclei pixels from _every_ mask
        final_mask = np.zeros(masks[0].shape)
        for m in masks:
            final_mask = np.logical_or(final_mask, m)
        final_mask = final_mask.astype(np.float32)

        # Not every image in the dataset is the same size.  To simplify the problem, we are going to re-size  every image to be (256, 256)
        img, final_mask = torch.tensor(img), torch.tensor(final_mask).unsqueeze(
            0)  # First we convert to PyTorch tensors

        # The interpolate function can be used to re-size a batch of images. So we make each image a "batch" of 1
        img = F.interpolate(img.unsqueeze(0), (256, 256))
        final_mask = F.interpolate(final_mask.unsqueeze(0), (256, 256))
        # Now the shapes  are (B=1, C, W, H) We need to convert them back to FloatTensors and grab the first item in the "batch". This will return a tuple of: (3, 256, 256), (1, 256, 256)
        return img.type(torch.FloatTensor)[0], final_mask.type(torch.FloatTensor)[0]


dsb_data = DSB2018(paths)

train_split, test_split = torch.utils.data.random_split(dsb_data, [90, len(dsb_data)-90])
train_seg_loader = DataLoader(train_split, batch_size=1, shuffle=True)
test_seg_loader = DataLoader(test_split,  batch_size=1)

loss_func = nn.BCEWithLogitsLoss()

class CnnNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 1, 3, 1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)

        return F.sigmoid(x)


num_epochs = 5

model = CnnNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(num_epochs):
    model.train()
    for x, y in train_seg_loader:
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_func(prediction, y)

        loss.backward()
        optimizer.step()

model.load_state_dict(torch.load('./cnn.pt'))

sample_data = test_split[6][0].unsqueeze(0)
prediction = model(sample_data)
prediction_image = prediction >= 0.5 * 1