from model import EfficientNet
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import random
import h5py
import ipdb
import openslide as osd
import torch
import torchvision
import random
from skimage.filters import threshold_multiotsu

import numpy as np

from torch.utils.data import Dataset
from PIL import Image


class DatasetLoader(Dataset):
    def __init__(self, h5paths):
        self.h5paths = h5paths
        # self.ff = h5py.File('/home/stat-huamenglei/PathFinder/PathFinder-main/WSI_decoupling/model_my/all_indexs.h5', "r")
        self.data = pd.read_csv('/home/stat-huamenglei/PathFinder/PathFinder-main/Data/clinical_information/GC_2300_all.csv')# all 加或不加
        self.death_time = self.data['death_time']
        self.death_status = self.data['death_status']
        self.wsi = self.data['WSIs']

    def __len__(self):
        return len(self.h5paths)


    def __getitem__(self, idx):
        path = self.h5paths[idx]
        # start_time = time.time()
        id = path.split('/')[-1][:-3]
        # print(id)
        f = h5py.File(path, "r")
        num = len(f['features'])
        index = sorted(random.sample(range(num),200))
        features = f['features'][index]

        # end_time = time.time()
        # print("运行时间：", end_time - start_time, "秒")
        
        return features



class CustomizedDataset_Fusion(Dataset):
    def __init__(self, path_set, csv_file, rs_tiles, nr_tiles, transform = None):
       
        self.path_set = path_set
        self.csv = pd.read_csv(csv_file)
        self.rs_tiles = rs_tiles
        self.nr_tiles = nr_tiles
        self.transform = transform
        # self.slideimage = osd.OpenSlide(path_set)
        self.imagesize = 256
        # self.non_image = non_image
        # self.encoding_scheme = encoding_scheme 
    
    def _test_cut(self,i,j,thresh_otsu,downsampled_size):
        if thresh_otsu.shape[1] >= (i+1)*downsampled_size:
            if thresh_otsu.shape[0] >= (j+1)*downsampled_size:
                cut_thresh = thresh_otsu[j*downsampled_size:(j+1)*downsampled_size, i*downsampled_size:(i+1)*downsampled_size]
            else:
                cut_thresh = thresh_otsu[(j)*downsampled_size:thresh_otsu.shape[0], i*downsampled_size:(i+1)*downsampled_size]
        else:
            if thresh_otsu.shape[0] >= (j+1)*downsampled_size:
                cut_thresh = thresh_otsu[j*downsampled_size:(j+1)*downsampled_size, (i)*downsampled_size:thresh_otsu.shape[1]]
            else:
                cut_thresh = thresh_otsu[(j)*downsampled_size:thresh_otsu.shape[0], (i)*downsampled_size:thresh_otsu.shape[1]]
        return cut_thresh

    def __len__(self):
        return len(self.path_set)
    
    def _get_images(self,slide_path): # get all tiles of the slides in path_set 
                        
        slideimage = osd.OpenSlide(slide_path)
        downsampling = slideimage.level_downsamples
        if len(downsampling) > 2:
            best_downsampling_level = 2
            downsampling_factor = int(self.slideimage.level_downsamples[best_downsampling_level])
            svs_native_levelimg = self.slideimage.read_region((0, 0), best_downsampling_level, self.slideimage.level_dimensions[best_downsampling_level])
            svs_native_levelimg = svs_native_levelimg.convert('L')
            img = np.array(svs_native_levelimg)

            # 基于 otsu 方法的返回灰度值区间
            thresholds = threshold_multiotsu(img)
            # 返回img在bins的位置，二值化
            regions = np.digitize(img, bins=thresholds)
            regions[regions == 1] = 0
            regions[regions == 2] = 1
            thresh_otsu = regions
            imagesize = self.imagesize
            downsampled_size = int(imagesize /downsampling_factor)
            Width = self.slideimage.dimensions[0]
            Height = self.slideimage.dimensions[1]
            num_row = int(Height/imagesize) + 1
            num_col = int(Width/imagesize) + 1

            
            samples = []
            for _ in range(self.nr_tiles):
                j = random.sample(range(0, num_row),1)[0]
                i = random.sample(range(0, num_col),1)[0]
                cut_thresh = self._test_cut(i,j,thresh_otsu,downsampled_size)
                while np.mean(cut_thresh) > 0.75:
                    j = random.sample(range(0, num_row),1)[0]
                    i = random.sample(range(0, num_row),1)[0]
                    cut_thresh = self._test_cut(i,j,thresh_otsu,downsampled_size)

                filter_location = (i*imagesize, j*imagesize)
                level = 0
                patch_size = (imagesize, imagesize)
                location = (filter_location[0], filter_location[1])
                
                CutImage = self.slideimage.read_region(location, level, patch_size)
                img_np = np.array(CutImage.convert('RGB'))
                samples.append(img_np)
        
        return samples
    
    def __getitem__(self, idx):
        """
        non_image parameters decides whether only images, or images and patient data are returned 
        """
        img_path = self.path_set[idx]
        samples = self._get_images(img_path) # thats a tile
        slide_name = img_path.split('/')[-1][:12]
        row_idx =  self.csv[self.csv['patient']== slide_name].index.tolist()[0]
        label = self.csv['label'][row_idx]
        
        
        # position assignment of the tile within the slide 



        if self.transform: 
            image = self.transform(image)
        
        # if self.non_image != None: 
        #     if self.encoding_scheme == 'scale01' or self.encoding_scheme == 'unscaled' or self.encoding_scheme == 'classified_2':
        #         non_image_data = torch.zeros(len(self.non_image)) 
        #         for counter, key in enumerate(self.non_image): 
        #             non_image_data[counter] = self.csv_files[key][row_idx]
                
        #     if self.encoding_scheme == 'onehot':
        #         non_image_data = []
        #         for key in self.non_image: 
        #             non_image_data.append(self.csv_files[key][row_idx])
        #         non_image_data = np.concatenate(non_image_data)
        #         non_image_data = torch.Tensor(non_image_data)
        #         #print(non_image_data)
                    
        #     sample = image, non_image_data, label
            
        # else: 
            # sample = image, label

        return image, label 
        
        
        
        
        

if __name__ == "__main__":
    # model_ft = EfficientNet.from_pretrained('efficientnet-b4', num_classes = 2)
    ipdb.set_trace()
    data = pd.read_csv('data.csv')
    slide_path_base = '/mnt/usb4/hml/TCGA-STAD'
    path_set = [os.path.join(slide_path_base,p) for p in os.listdir(slide_path_base)]
    train_path_set = path_set                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    train_dataset = CustomizedDataset_Fusion(path_set)



