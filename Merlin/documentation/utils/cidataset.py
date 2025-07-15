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

class CiDataset(data.Dataset):#需要继承data.Dataset
    def __init__(self,img_path,annotation_path, exfeatures_path, name):
        # 1. Initialize file path or list of file names.\
        self.flag = 1 # 1 for 12 month or 2 for last
        self.threshold1 = 38
        self.threshold2 = 100
        self.name = name # "train" or "validation"
        self.imgpaths, self.l_or_r, self.labels, self.ex_features = self.__make_dataset(img_path, annotation_path, exfeatures_path)
        # self.input_images = self.__get_images()
        # print(self.imgpaths, self.l_or_r, self.labels, self.ex_features)
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
        folder_pth = self.imgpaths[index]
        image_data = []

        image = sitk.GetArrayFromImage(sitk.ReadImage(folder_pth + "_0000.nii.gz"))
        image = image[np.newaxis, :].transpose(0, 2, 3, 1)
        # image = self.input_images[index]
        # print(image.shape)
        # image = image.transpose(0, 3, 1, 2)
        # print(image.shape)

        # if self.l_or_r[index] == "l":
        #     for file in os.listdir(folder_pth):
        #         image = cv2.imread(os.path.join(folder_pth, file), cv2.IMREAD_GRAYSCALE)
        #         image_data.append(np.array(image)[256-64+32+randx:256+128-32+randx,256-128+randy:256+randy])
        # else:
        #     for file in os.listdir(folder_pth):
        #         image = cv2.imread(os.path.join(folder_pth, file), cv2.IMREAD_GRAYSCALE)
        #         image_data.append(np.array(image)[256-64+32+randx:256+128-32+randx,256+randy:256+128+randy])
        if self.l_or_r[index] == "l":
            image_data = image[:, 256 - 32: 512 - 128 - 32, 256 - 128: 256, :]
        else:
            image_data = image[:, 256 - 32: 512 - 128 - 32, 256: 256 + 128, :]
        # if self.l_or_r[index] == "l":
        #     image_data = image[:, 256: 512 - 128, 256 - 128 + 32: 256 + 32, :]
        # else:
        #     image_data = image[:, 256: 512 - 128, 256 - 32: 256 + 128 - 32, :]

        # for i in range(64):
        #     image_path = f"outputs/image_{i}.png"
        #     cv2.imwrite(image_path, image_data[0,:,:,i])
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
            # alpha = random.uniform(0.8, 1.2)
            # beta = random.uniform(-16, 16)
            # temp = tfs(img3d).numpy()
            # for i in range(64):
            #     image_path = f"outputs/image_{i}.png"
            #     cv2.imwrite(image_path, temp[0, :, :, i])
            return  tfs(img3d).numpy().transpose(0, 3, 1, 2) / 255.0, self.labels[index], np.concatenate((self.ex_features[index][0:30], self.ex_features[index][45:47])).astype("float32")
            # return  np.clip(tfs(img3d).numpy() * alpha + beta, 0, 255) / 255.0, self.labels[index]
        else:
            return rawdata.transpose(0, 3, 1, 2) / 255.0, self.labels[index], np.concatenate((self.ex_features[index][0:30], self.ex_features[index][45:47])).astype("float32")

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        # print(len(self.imgpaths), len(self.l_or_r), len(self.labels))
        return len(self.labels)
    
    def __get_images(self):
        input_images = []
        for ind, folder_pth in enumerate(self.imgpaths):
            image = nib.load(folder_pth + "_0000.nii.gz").get_fdata()
            mask_1 = nib.load(folder_pth.replace("imagesTs", "mask1") + ".nii.gz").get_fdata()
            mask_2 = nib.load(folder_pth.replace("imagesTs", "mask2") + ".nii.gz").get_fdata()
            mask_3 = nib.load(folder_pth.replace("imagesTs", "mask3") + ".nii.gz").get_fdata()
            # print(image.shape, mask_1.shape, mask_2.shape, mask_3.shape)
            channel1 = image * mask_1
            channel2 = image * mask_2
            channel3 = image * mask_3
            image = np.array([image, channel1, channel2, channel3])
            input_images.append(image)

        return input_images


    def __make_dataset(self, img_path, annotation_path, exfeatures_path):
        df = pd.read_excel(annotation_path, dtype={'name': str})
        ex_features_df = pd.read_csv(exfeatures_path, dtype={'name': str})
        path = []
        labels = []
        l_or_r = []
        ex_features = []
        assert self.flag in [1, 2]
        for index, row in df.iterrows():
            assert row['lr'] in ['双', '左侧', '右侧']
            folder_pth = os.path.join(img_path, row['name'])
            if row['lr'] == '双':
                if self.flag == 1:
                    if not pd.isna(row['1_0.5k']):
                        avg = row['1_0.5k'] + row['1_1k'] + row['1_2k'] + row['1_4k']
                        avg = avg / 4
                        if avg < self.threshold1:
                            labels.append(0)
                        elif avg >= self.threshold1 and avg < self.threshold2:
                            labels.append(1)
                        else:
                            labels.append(2)
                        # labels.append(avg)
                        path.append(folder_pth)
                        l_or_r.append("l")
                        ex_features.append(ex_features_df.loc[ex_features_df['name'] == row['name'] + 'l'].drop(columns=['name'], inplace=False).values[0])
                else:
                    if not pd.isna(row['3_0.5k']):
                        avg = row['3_0.5k'] + row['3_1k'] + row['3_2k'] + row['3_4k']
                        avg = avg / 4
                        if avg < self.threshold1:
                            labels.append(0)
                        elif avg >= self.threshold1 and avg < self.threshold2:
                            labels.append(1)
                        else:
                            labels.append(2)
                        # labels.append(avg)
                        path.append(folder_pth)
                        l_or_r.append("l")
                        ex_features.append(ex_features_df.loc[ex_features_df['name'] == row['name'] + 'l'].drop(columns=['name'], inplace=False).values[0])
                if self.flag == 1:
                    if not pd.isna(row['2_0.5k']):
                        avg = row['2_0.5k'] + row['2_1k'] + row['2_2k'] + row['2_4k']
                        avg = avg / 4
                        if avg < self.threshold1:
                            labels.append(0)
                        elif avg >= self.threshold1 and avg < self.threshold2:
                            labels.append(1)
                        else:
                            labels.append(2)
                        # labels.append(avg)
                        path.append(folder_pth)
                        l_or_r.append("r")
                        ex_features.append(ex_features_df.loc[ex_features_df['name'] == row['name'] + 'r'].drop(columns=['name'], inplace=False).values[0])
                else:
                    if not pd.isna(row['4_0.5k']):
                        avg = row['4_0.5k'] + row['4_1k'] + row['4_2k'] + row['4_4k']
                        avg = avg / 4
                        if avg < self.threshold1:
                            labels.append(0)
                        elif avg >= self.threshold1 and avg < self.threshold2:
                            labels.append(1)
                        else:
                            labels.append(2)
                        # labels.append(avg)
                        path.append(folder_pth)
                        l_or_r.append("r")
                        ex_features.append(ex_features_df.loc[ex_features_df['name'] == row['name'] + 'r'].drop(columns=['name'], inplace=False).values[0])
            elif row['lr'] == '左侧':
                if self.flag == 1:
                    if not pd.isna(row['1_0.5k']):
                        avg = row['1_0.5k'] + row['1_1k'] + row['1_2k'] + row['1_4k']
                        avg = avg / 4
                        if avg < self.threshold1:
                            labels.append(0)
                        elif avg >= self.threshold1 and avg < self.threshold2:
                            labels.append(1)
                        else:
                            labels.append(2)
                        # labels.append(avg)
                        path.append(folder_pth)
                        l_or_r.append("l")
                        ex_features.append(ex_features_df.loc[ex_features_df['name'] == row['name'] + 'l'].drop(columns=['name'], inplace=False).values[0])
                else:
                    if not pd.isna(row['3_0.5k']):
                        avg = row['3_0.5k'] + row['3_1k'] + row['3_2k'] + row['3_4k']
                        avg = avg / 4
                        if avg < self.threshold1:
                            labels.append(0)
                        elif avg >= self.threshold1 and avg < self.threshold2:
                            labels.append(1)
                        else:
                            labels.append(2)
                        # labels.append(avg)
                        path.append(folder_pth)                       
                        l_or_r.append("l")
                        ex_features.append(ex_features_df.loc[ex_features_df['name'] == row['name'] + 'l'].drop(columns=['name'], inplace=False).values[0])
            else:
                if self.flag == 1:
                    if not pd.isna(row['2_0.5k']):
                        avg = row['2_0.5k'] + row['2_1k'] + row['2_2k'] + row['2_4k']
                        avg = avg / 4
                        if avg < self.threshold1:
                            labels.append(0)
                        elif avg >= self.threshold1 and avg < self.threshold2:
                            labels.append(1)
                        else:
                            labels.append(2)
                        # labels.append(avg)
                        path.append(folder_pth)
                        l_or_r.append("r")
                        ex_features.append(ex_features_df.loc[ex_features_df['name'] == row['name'] + 'r'].drop(columns=['name'], inplace=False).values[0])
                else:
                    if not pd.isna(row['4_0.5k']):
                        avg = row['4_0.5k'] + row['4_1k'] + row['4_2k'] + row['4_4k']
                        avg = avg / 4
                        if avg < self.threshold1:
                            labels.append(0)
                        elif avg >= self.threshold1 and avg < self.threshold2:
                            labels.append(1)
                        else:
                            labels.append(2)
                        # labels.append(avg)
                        path.append(folder_pth)
                        l_or_r.append("r")
                        ex_features.append(ex_features_df.loc[ex_features_df['name'] == row['name'] + 'r'].drop(columns=['name'], inplace=False).values[0])
        return path, l_or_r, labels, ex_features


if __name__ == '__main__':
    dataset = CiDataset("/mnt/data/wwx/data/imagesTs", "/mnt/data/wwx/data/label.xlsx", "/mnt/data/wwx/data/scaled_features.csv", "train")
    # a = dataset[0]
    import datetime
    current_time = datetime.datetime.now()
    # print(current_time)
    # print(len(dataset))
    ind = 50
    print(dataset[ind][0].shape)
    print(dataset.imgpaths[ind])
    print(dataset.l_or_r[ind])
    print(dataset.ex_features[ind])
    # for i in range(0, len(dataset)):
    #     print(i, dataset[i][0].shape)
    #     current_time = datetime.datetime.now()
    #     print(current_time)
