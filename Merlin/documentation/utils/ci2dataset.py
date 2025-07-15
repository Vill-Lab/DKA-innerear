from torch.utils import data
import numpy as np
import os
import pandas as pd
import cv2
from torchvision.transforms import transforms
import torch
from PIL import Image
import random
import torchio as tio
import nibabel as nib
import SimpleITK as sitk

class Ci2Dataset(data.Dataset):#需要继承data.Dataset
    def __init__(self, img_path, annotation_path, exfeatures_path, name):
        # 1. Initialize file path or list of file names.\

        self.name = name # "train" or "validation"
        self.imgpths, self.l_or_rs, self.labels, self.ex_features = self.__make_dataset(img_path, annotation_path, exfeatures_path)
        # self.input_images = self.__get_images()

        # print(self.imgpths, self.l_or_rs, self.labels)
        # exit()

    def getlabels(self):
        return self.labels
    
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        # itempath = os.path.join('..', self.rootpath, self.datapath[index])
        # itemdata
        # print(index, index >= len(self.data))
        # print(len(self.data[index]))
        folder_pth = self.imgpths[index]
        image_data = []
        image = sitk.GetArrayFromImage(sitk.ReadImage(folder_pth + ".nii.gz"))
        image = image[np.newaxis, :].transpose(0, 2, 3, 1)

        if self.l_or_rs[index] == "r":
            image_data = image[:, 256 - 32: 512 - 128 - 32, 256 - 128: 256, :]
        else:
            image_data = image[:, 256 - 32: 512 - 128 - 32, 256: 256 + 128, :]
        # if self.l_or_r[index] == "l":
        #     image_data = image[:, 256: 512 - 128, 256 - 128 + 32: 256 + 32, :]
        # else:
        #     image_data = image[:, 256: 512 - 128, 256 - 32: 256 + 128 - 32, :]

        for i in range(64):
            image_path = f"outputs/image_{i}.png"
            cv2.imwrite(image_path, image_data[0,:,:,i])
        # Save the image to the desired location

        # rawdata = np.expand_dims(np.array(image_data, dtype="float32"), axis=0)
        rawdata = np.array(image_data, dtype="float32")
        if self.name == "train":
            img3d = tio.Image(tensor=torch.from_numpy(rawdata))
            tfs = tio.transforms.Compose([
                tio.transforms.RandomFlip(1, 0.5),
                tio.transforms.RandomAffine(
                    scales=(0.8, 1.2, 0.8, 1.2, 1.0, 1.0),
                    degrees=10,
                    translation=(16, 16, 0),
                    center='image',
                    image_interpolation="bspline"
                ),
                tio.transforms.RandomGamma(0.05)
            ])
            # tfs = tio.transforms.Compose([
            #     tio.transforms.RandomFlip(1, 0.5),
            #     tio.transforms.RandomAffine(
            #         scales=(0.9, 1.1, 0.9, 1.1, 1.0, 1.0),
            #         degrees=5,
            #         translation=(8, 8, 0),
            #         center='image',
            #         image_interpolation="bspline"
            #     ),
            #     tio.transforms.RandomGamma(0.05)
            # ])
            # alpha = random.uniform(0.8, 1.2)
            # beta = random.uniform(-16, 16)
            # temp = tfs(img3d).numpy()
            # for i in range(64):
            #     image_path = f"outputs/image_{i}.png"
            #     cv2.imwrite(image_path, temp[0, :, :, i])
            return  tfs(img3d).numpy().transpose(0, 3, 1, 2) / 255.0, self.labels[index].to_numpy(dtype="float32"), self.ex_features[index].astype("float32")
            # return  tfs(img3d).numpy().transpose(0, 3, 1, 2) / 255.0, np.array(1 if self.labels[index].any() else 0)
            # return  np.clip(tfs(img3d).numpy() * alpha + beta, 0, 255) / 255.0, self.labels[index]
        else:
            return rawdata.transpose(0, 3, 1, 2) / 255.0, self.labels[index].to_numpy(dtype="float32"), self.ex_features[index].astype("float32")
            # return rawdata.transpose(0, 3, 1, 2) / 255.0, np.array(1 if self.labels[index].any() else 0)

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        # print(len(self.imgpaths), len(self.l_or_r), len(self.labels))
        return len(self.labels)
    
    def __make_dataset(self, img_path, annotation_path, exfeatures_path):
        df = pd.read_csv(annotation_path, dtype={'name': str})
        ex_features_df = pd.read_csv(exfeatures_path, dtype={'name': str})

        pths = []
        l_or_rs = []
        labels = []
        ex_features = []

        for index, row in df.iterrows():
            pth = os.path.join(img_path, row['name'][:-1])
            if os.path.exists(pth + ".nii.gz"):
                pths.append(pth)
                l_or_rs.append(row['name'][-1:])
                labels.append(row[['label1','label2','label3','label4','label5','label6']])
                ex_features.append(ex_features_df.loc[ex_features_df['name'] == row['name']].drop(columns=['name'], inplace=False).values[0])

        return pths, l_or_rs, labels, ex_features


if __name__ == '__main__':
    dataset = Ci2Dataset("/media/ubuntu/VILL1/wwx/FinalData2", "/media/ubuntu/VILL1/wwx/FinalData2/ci2label.csv", "/media/ubuntu/VILL1/wwx/FinalData2/scaled_features.csv", "train")
    import datetime
    current_time = datetime.datetime.now()
    ind = 29
    print(dataset[ind][0].shape)
    print(dataset[ind][1])
    print(dataset[ind][2])
    print(dataset[ind][2].shape)
    print("wwxnb")
    print(type(dataset.labels[ind]))
    print(dataset.labels[ind])
    print(dataset.imgpths[ind])
    print(dataset.l_or_rs[ind])
    # for i in range(0, len(dataset)):
    #     print(i, dataset[i][0].shape)
    #     current_time = datetime.datetime.now()
    #     print(current_time)    
    # Initialize counters for each label
    zero_counts = np.zeros(6, dtype=int)
    one_counts = np.zeros(6, dtype=int)

    for i in range(len(dataset)):
        labels = dataset.getlabels()[i].to_numpy(dtype="int32")
        for j in range(6):
            zero_counts[j] += np.sum(labels[j] == 0)
            one_counts[j] += np.sum(labels[j] == 1)

    for j in range(6):
        print(f"Label {j+1}:")
        print(f"total: {zero_counts[j] + one_counts[j]}")
        print(f"  Number of 0s: {zero_counts[j]}")
        print(f"  Number of 1s: {one_counts[j]}")
    
    zero_counts = 0
    one_counts = 0
    for i in range(len(dataset)):
        labels = dataset.getlabels()[i]
        if labels.any():
            one_counts += 1
        else:
            zero_counts += 1
    print(f"total: {zero_counts + one_counts}")
    print(f"  Number of 0s: {zero_counts}")
    print(f"  Number of 1s: {one_counts}")

