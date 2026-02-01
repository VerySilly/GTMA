#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:10:37 2021

@author: kyungsub
"""
import spams
import numpy as np
import cv2
import time
import torch.nn as nn
from sklearn.cluster import KMeans
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
import openslide as osd
from torchvision import transforms
from torch_geometric.data import Data
from EfficientNet.model import EfficientNet
import superpatch_network_construction 
# import false_graph_filterinpthg
from torch.utils.data import Dataset, DataLoader
from skimage.filters import threshold_multiotsu
import pickle
import argparse
import ipdb
from glob import glob
import staintools
from PIL import Image
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import multiprocessing as mp

# from vahadane import vahadane

def prepare_normalizer(target_image_path):
    """Prepare stain normalizer using a target image."""
    target = staintools.read_image(target_image_path)
    target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target)
    return normalizer

def process_image(img, vhd, model):
    Ws, Hs = vhd.stain_separate(img)
    model.fit(Hs.T)
    center = model.cluster_centers_
    return {'Ws': Ws.reshape(-1), 'Hs': Hs.reshape(-1), 'center': center.reshape(-1)}

def _process_batch(imgs, vhd, model):
    # imgs = sample_img['img']
    results = []
    with mp.Pool(processes=12) as pool:
        results = pool.starmap(process_image, [(np.clip(np.array(img) * 255.0 / np.percentile(np.array(img), 90), 0, 255).astype(np.uint8), vhd, model) for img in imgs])
        # futures = [executor.submit(process_image, np.clip(np.array(img) * 255.0 / np.percentile(np.array(img), 90), 0, 255).astype(np.uint8), vhd, model) for img in imgs]
        # for future in futures:
        #     results.append(future.result())
    return results


def process_batch(imgs, vhd, model):
    results = []
    for img in imgs:
        img = np.array(img)
        p = np.percentile(img, 90)
        img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)
        Ws, Hs = vhd.stain_separate(img)
        model.fit(Hs.T)
        center = model.cluster_centers_
        results.append({
            'Ws': Ws.reshape(-1),
            'Hs': Hs.reshape(-1),
            'center': center.reshape(-1)
        })
    return results

def local_binary_pattern_hist(img_imp):
    """
    Calculate the local binary pattern of the given input.
    Input:
        img_imp (PIL.Image): The input grey scale image represented.
    Output:
        hist (np.array):The histogram of the local binary pattern of input image.
    """
    lbp = local_binary_pattern(img_imp, 8, 1, 'ror')
    hist = np.histogram(lbp, density=True, bins=128, range=(0, 128))[0]
    return hist




def pre_filtering(img):
    """
    Filter out the white region and calculate the rgb/lbp histogram for a patch in the given slide.
    Input:
        slide_name (str): The slide to process
        coord (np.array): The coordinate of the patch in the slide
        patch_size (int): The height and width of the patch
    Output:
        hist_feat (np.array): RGB histogram of patch in coord from the slide
        lbp_feat (np.array): LBP histogram of patch in the coord from the slide
    """
    hist_feat = []
    # wsi = openslide.open_slide(slide_name)
    # patch = wsi.read_region((coord[0], coord[1]), 0, (patch_size, patch_size))

    # Convert to 5x to do filtering
    patch_grey = img.convert('L')
    _, white_region = cv2.threshold(np.array(patch_grey), 235, 255, cv2.THRESH_BINARY)
    if np.sum(white_region == 255) / (256 * 256) > 0.9:
        return None, None

    # Convert to 5x to extract RGB histogram
    patch_rgb = img.convert("RGB").resize((256, 256))
    patch_rgb = np.array(patch_rgb).astype('float32')
    for i, col in enumerate(('r', 'g', 'b') ):
        histr = cv2.calcHist([patch_rgb], [i], None, [256], [0, 256])
        hist_feat.append(histr.T)
    hist_feat = np.concatenate(hist_feat, 1)

    lbp_feat = local_binary_pattern_hist(np.array(patch_grey))
    return hist_feat, lbp_feat

class vahadane(object):
    
    def __init__(self, STAIN_NUM=2, THRESH=0.9, LAMBDA1=0.01, LAMBDA2=0.01, ITER=100, fast_mode=0, getH_mode=0):
        self.STAIN_NUM = STAIN_NUM
        self.THRESH = THRESH
        self.LAMBDA1 = LAMBDA1
        self.LAMBDA2 = LAMBDA2
        self.ITER = ITER
        self.fast_mode = fast_mode # 0: normal; 1: fast
        self.getH_mode = getH_mode # 0: spams.lasso; 1: pinv;


    def show_config(self):
        print('STAIN_NUM =', self.STAIN_NUM)
        print('THRESH =', self.THRESH)
        print('LAMBDA1 =', self.LAMBDA1)
        print('LAMBDA2 =', self.LAMBDA2)
        print('ITER =', self.ITER)
        print('fast_mode =', self.fast_mode)
        print('getH_mode =', self.getH_mode)


    def getV(self, img):
        
        I0 = img.reshape((-1,3)).T
        I0[I0==0] = 1

        V0 = np.log(255 / I0)

        img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        mask = img_LAB[:, :, 0] / 255 < self.THRESH
        I = img[mask].reshape((-1, 3)).T
        I[I == 0] = 1
        V = np.log(255 / I)
        return V0, V


    def getW(self, V):
        W = spams.trainDL(np.asfortranarray(V), K=self.STAIN_NUM, lambda1=self.LAMBDA1, iter=self.ITER, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False)
        W = W / np.linalg.norm(W, axis=0)[None, :]
        if (W[0,0] < W[0,1]):
            W = W[:, [1,0]]
        return W


    def getH(self, V, W):
        if (self.getH_mode == 0):
            H = spams.lasso(np.asfortranarray(V), np.asfortranarray(W), mode=2, lambda1=self.LAMBDA2, pos=True, verbose=False).toarray()
        elif (self.getH_mode == 1):
            H = np.linalg.pinv(W).dot(V)
            H[H<0] = 0
        else:
            H = 0
        return H


    def stain_separate(self, img):
        start = time.time()
        if (self.fast_mode == 0):
            V0, V = self.getV(img)
            W = self.getW(V)
            H = self.getH(V0, W)
        elif (self.fast_mode == 1):
            m = img.shape[0]
            n = img.shape[1]
            grid_size_m = int(m / 5)
            lenm = int(m / 20)
            grid_size_n = int(n / 5)
            lenn = int(n / 20)
            W = np.zeros((81, 3, self.STAIN_NUM)).astype(np.float64)
            for i in range(0, 4):
                for j in range(0, 4):
                    px = (i + 1) * grid_size_m
                    py = (j + 1) * grid_size_n
                    patch = img[px - lenm : px + lenm, py - lenn: py + lenn, :]
                    # print('{},{},{}'.format(i, j, patch.shape))
                    V0, V = self.getV(patch)
                    W[i*9+j] = self.getW(V)
            W = np.mean(W, axis=0)
            V0, V = self.getV(img)
            H = self.getH(V0, W)
        print('stain separation time:', time.time()-start, 's')
        return W, H


    def SPCN(self, img, Ws, Hs, Wt, Ht):
        Hs_RM = np.percentile(Hs, 99)
        Ht_RM = np.percentile(Ht, 99)
        Hs_norm = Hs * Ht_RM / Hs_RM
        Vs_norm = np.dot(Wt, Hs_norm)
        Is_norm = 255 * np.exp(-1 * Vs_norm)
        # print(Is_norm.shape)
        I = Is_norm.T.reshape(img.shape).astype(np.uint8)
        return I


class SurvivalImageDataset():

    """
    Target dataset has the list of images such as
    _patientID_SurvDay_Censor_TumorStage_WSIPos.tif
    """

    def __init__(self, image, x, y, transform,normalizer):
        
 
        self.vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=0, getH_mode=0, ITER=50)
        self.model = KMeans(n_clusters=2, random_state=0)
        self.image = image
        self.x = x
        self.y = y
        self.transform = transform
        self.normalizer = normalizer

    def __len__(self):
        return len((self.image))

    def _read_image(self,img):
        
        img = np.array(img)
        p = np.percentile(img, 90)
        img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)
        return img
    




    def __getitem__(self, idx):

        """
        patientID, SurvivalDuration, SurvivalCensor, Stage,
        ProgressionDuration, ProgressionCensor, MetaDuration, MetaCensor
        """
        transform = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
                ])
        #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        image = self.image[idx]
        x = self.x[idx]
        y = self.y[idx]
        image = image.convert('RGB')
        # ipdb.set_trace()

        # 颜色标准化
        
        # image = self.normalizer.transform(image)
        # image = Image.fromarray(image)



        R = transform(image)
        # img = self._read_image(image)

        # # ipdb.set_trace()
        # Ws,Hs = self.vhd.stain_separate(img)
        # self.model.fit(Hs.T)
        # center = self.model.cluster_centers_
        image = np.array(image)
        sample = { 'img' : image, 'image' : R,'X' : torch.tensor(x), 'Y' : torch.tensor(y)}
    
        return sample



def supernode_generation(image, model_ft,clf, device, Argument, save_dir):

    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    origin_dir = os.path.join(save_dir, 'original')
    if os.path.exists(origin_dir) is False:
        os.mkdir(origin_dir)

    superpatch_dir = os.path.join(save_dir, 'superpatch')
    if os.path.exists(superpatch_dir) is False:
        os.mkdir(superpatch_dir)

    transform = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
                ])
    
    
    
    normalizer = prepare_normalizer('/home/stat-huamenglei/1404814_105.png')
    
    
    
    threshold = Argument.threshold
    spatial_threshold = Argument.spatial_threshold
    
    sample = image.split('/')[-1][:-4]
    
    
    image_path = image
    try:
        slideimage = osd.OpenSlide(image_path)
    except:
        print('openslide error')
        return 0
    downsampling = slideimage.level_downsamples

    if len(downsampling) > 2:
        # imagesize=256
        imagesize = Argument.imagesize
        if 'aperio.AppMag' in slideimage.properties.keys():
            level_0_magnification = int(slideimage.properties['aperio.AppMag'])
        elif 'openslide.mpp-x' in slideimage.properties.keys():
            level_0_magnification = 40 if int(np.floor(float(slideimage.properties['openslide.mpp-x']) * 10)) == 2 else 20
        else:
            print('没缩放比例')
            pass
            # level_0_magnification = 40
        best_downsampling_level = 2
        downsampling_factor = int(slideimage.level_downsamples[best_downsampling_level])
        if level_0_magnification == 40:
            cut_size = imagesize*2
        elif level_0_magnification == 20:
            cut_size = imagesize
        else:
            print(f'level_0_magnification:{level_0_magnification}')
        print('{}进入函数,downsampling:{},最大放大倍数{}'.format(sample[:12],downsampling,level_0_magnification))
        # ipdb.set_trace()
        if len(downsampling) > 2:
            # print(f'最大放大倍数： {level_0_magnification}')
            # Get the image at the requested scale
            svs_native_levelimg = slideimage.read_region((0, 0), best_downsampling_level, slideimage.level_dimensions[best_downsampling_level])
            

            
            
            
            svs_native_levelimg = svs_native_levelimg.convert('L')
            img = np.array(svs_native_levelimg)

            # 基于 otsu 方法的返回灰度值区间
            thresholds = threshold_multiotsu(img)
            # 返回img在bins的位置，二值化
            regions = np.digitize(img, bins=thresholds)
            regions[regions == 1] = 0
            regions[regions == 2] = 1
            thresh_otsu = regions

            
            downsampled_size = int(cut_size /downsampling_factor)
            Width = slideimage.dimensions[0]
            Height = slideimage.dimensions[1]
            num_row = int(Height/(cut_size)) + 1
            num_col = int(Width/(cut_size)) + 1
            x_list = []
            y_list = []
            feature_list = []
            x_y_list = []
            counter = 0
            inside_counter = 0
            temp_patch_list = []
            temp_x = []
            temp_y = []
            rgb_list = []
            lbp_list = []
            

            # 生成 feature_list,x_y_list
            with tqdm(total = num_row * num_col) as pbar_image:
                for i in range(0, num_col):
                    for j in range(0, num_row):

                        # 获得cut_thresh
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
                                


                        # 如果 cut_thresh大于0.75 直接不储存
                        if np.mean(cut_thresh) > 0.75:
                            pbar_image.update()
                            pass
                        else:
                            if level_0_magnification == 40:
                                
                                filter_location = (i*imagesize*2, j*imagesize*2)                        
                                level = 0
                                patch_size = (imagesize*2, imagesize*2)
                                location = (filter_location[0], filter_location[1])
                            
                                CutImage = slideimage.read_region(location, level, patch_size)
                                CutImage=CutImage.resize((imagesize, imagesize))
                            elif level_0_magnification == 20:
                                filter_location = (i*imagesize, j*imagesize)
                                level = 0
                                patch_size = (imagesize, imagesize)
                                location = (filter_location[0], filter_location[1])
                                CutImage = slideimage.read_region(location, level, patch_size)
                            else:
                                pass
                            
                            
                            hist_feat, lbp_feat = pre_filtering(CutImage)
                            rgb_list.append(hist_feat.reshape(-1))
                            lbp_list.append(lbp_feat)
                            temp_patch_list.append(CutImage)
                            x_list.append(i)
                            y_list.append(j)
                            temp_x.append(i)
                            temp_y.append(j)
                            counter += 1
                            batchsize = 128
                                
                                # 有效图片个数满足一个batch的数量时进行
                            if counter == batchsize:
                                # ipdb.set_trace()
                                # trash_pred = clf.predict(lbp_list)
                                # lbp = np.array(lbp_list)[trash_pred == 0]
                                # rgb = np.array(rgb_list)[trash_pred == 0]
                                # temp_patch_list = [temp_patch_list[i] for i in np.where(trash_pred == 0)[0]]
                                Dataset = SurvivalImageDataset(temp_patch_list, temp_x, temp_y, transform,normalizer)
                                dataloader = torch.utils.data.DataLoader(Dataset,batch_size=batchsize,num_workers=4,drop_last=False)
                                # vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=0, getH_mode=0, ITER=50)
                                # model = KMeans(n_clusters=2, random_state=0)
                                for sample_img in dataloader:
                                    images = sample_img['image']
                                    ipdb.set_trace()
                                    images = images.to(device)
                                    # results = process_batch(sample_img['img'], vhd, model)
                                    results = _process_batch(sample_img['img'], vhd, model)
                                    with torch.set_grad_enabled(False):
                                        # 类别是2分类，原文是自己在TCGA中训练过分辨正常组织和肿瘤组织
                                        classifier, features = model_ft(images)
                                        ipdb.set_trace()

                                # 生成feature_list，x_y_list 特征和坐标的list
                                if inside_counter == 0:
                                    feature_list = np.concatenate((features.cpu().detach().numpy(),
                                                                classifier.cpu().detach().numpy()), axis=1)
                                    temp_x = np.reshape(np.array(temp_x), (len(temp_x),1))
                                    temp_y = np.reshape(np.array(temp_y), (len(temp_x),1))
                                    
                                    x_y_list = np.concatenate((temp_x,temp_y),axis=1)
                                else:
                                    feature_list = np.concatenate((feature_list, 
                                                                np.concatenate((features.cpu().detach().numpy(),
                                                                                classifier.cpu().detach().numpy()),axis=1)), axis=0)
                                    temp_x = np.reshape(np.array(temp_x), (len(temp_x),1))
                                    temp_y = np.reshape(np.array(temp_y), (len(temp_x),1))
                                                
                                    x_y_list = np.concatenate((x_y_list, 
                                                            np.concatenate((temp_x,temp_y),axis=1)), axis=0)
                                inside_counter += 1
                                temp_patch_list = []
                                temp_x = []
                                temp_y = []
                                rgb_list = []
                                lbp_list = []
                                counter = 0
                                
                            pbar_image.update()

                # 整张片子没有batch数量的patch
                if counter < batchsize and counter >0:

                    trash_pred = clf.predict(lbp_list)
                    lbp_list = lbp_list[trash_pred == 0]
                    rgb_list = rgb_list[trash_pred == 0]
                    temp_patch_list = temp_patch_list[trash_pred == 0]
                    Dataset = SurvivalImageDataset(temp_patch_list, temp_x, temp_y, transform,normalizer)
                    dataloader = torch.utils.data.DataLoader(Dataset,batch_size=batchsize,num_workers=0,drop_last=False)
                    
                    for sample_img in dataloader:
                        images = sample_img['image']
                        images = images.to(device)
                        with torch.set_grad_enabled(False):
                            classifier, features = model_ft(images)
        
                        feature_list = np.concatenate((feature_list, 
                                                    np.concatenate((features.cpu().detach().numpy(),
                                                                    classifier.cpu().detach().numpy()),axis=1)), axis=0)
                        temp_x = np.reshape(np.array(temp_x), (len(temp_x),1))
                        temp_y = np.reshape(np.array(temp_y), (len(temp_x),1))
                                    
                        x_y_list = np.concatenate((x_y_list, 
                                                np.concatenate((temp_x,temp_y),axis=1)), axis=0)
                    temp_patch_list = []
                    temp_x = []
                    temp_y = []
                    rgb_list = []
                    lbp_list = []
                    counter = 0
            

            # 储存 特征、坐标
            feature_df = pd.DataFrame.from_dict(feature_list)
            coordinate_df = pd.DataFrame({'X': x_y_list[:,0],'Y': x_y_list[:,1]})
            graph_dataframe = pd.concat([coordinate_df, feature_df], axis = 1)
            graph_dataframe = graph_dataframe.sort_values(by = ['Y', 'X'])
            graph_dataframe = graph_dataframe.reset_index(drop = True)
            coordinate_df = graph_dataframe.iloc[:,0:2]
            feature_df.to_csv(os.path.join(origin_dir, sample + '_feature_list.csv'))
            coordinate_df.to_csv(os.path.join(origin_dir, sample+'_node_location_list.csv'))
            index = list(graph_dataframe.index)
            graph_dataframe.insert(0,'index_orig', index)
            
            node_dict = {}
            
            for i in range(len(coordinate_df)):
                node_dict.setdefault(i,[])
            
            X = max(set(np.squeeze(graph_dataframe.loc[:, ['X']].values,axis = 1)))
            Y = max(set(np.squeeze(graph_dataframe.loc[:, ['Y']].values, axis = 1)))
            
            # 删除对象
            del feature_df 



            
            gridNum = 4
            X_size = int(X / gridNum)
            Y_size = int(Y / gridNum)
        
            # 为什么size是÷4得到，循环使用的是6
            # 6*6个方格内
            # 获得  X_10      
            with tqdm(total=(gridNum+2)*(gridNum+2)) as pbar:
                for p in range(gridNum+2):
                    for q in range(gridNum+2):
                        if p == 0 :
                            if q == 0:
                                is_X = graph_dataframe['X'] <= X_size * (p+1)
                                is_X2 = graph_dataframe['X'] >= 0
                                is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                                is_Y2 = graph_dataframe['Y'] >= 0
                                X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                                
                            elif q == (gridNum+1):
                                is_X = graph_dataframe['X'] <= X_size * (p+1)
                                is_X2 = graph_dataframe['X'] >= 0
                                is_Y = graph_dataframe['Y'] <= Y
                                is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                                X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                                
                            else:
                                is_X = graph_dataframe['X'] <= X_size * (p+1)
                                is_X2 = graph_dataframe['X'] >= 0
                                is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                                is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2) # 为啥要减2？
                                X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                        elif p == (gridNum+1) :
                            if q == 0:
                                is_X = graph_dataframe['X'] <= X
                                is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                                is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                                is_Y2 = graph_dataframe['Y'] >= 0
                                X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                            elif q == (gridNum+1):
                                is_X = graph_dataframe['X'] <= X
                                is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                                is_Y = graph_dataframe['Y'] <= Y
                                is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                                X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                            else:
                                is_X = graph_dataframe['X'] <= X
                                is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                                is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                                is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                                X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                        else :
                            if q == 0:
                                is_X = graph_dataframe['X'] <= X_size * (p+1)
                                is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                                is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                                is_Y2 = graph_dataframe['Y'] >= 0
                                X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                            elif q == (gridNum+1):
                                is_X = graph_dataframe['X'] <= X_size * (p+1)
                                is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                                is_Y = graph_dataframe['Y'] <= Y
                                is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                                X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                            else:
                                is_X = graph_dataframe['X'] <= X_size * (p+1)
                                is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                                is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                                is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                                X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                        
                        if len(X_10) == 0:
                            pbar.update()
                            continue
                        
                        coordinate_dataframe = X_10.loc[:, ['X','Y']]
                        X_10 = X_10.reset_index(drop = True)
                        coordinate_list = coordinate_dataframe.values.tolist()
                        index_list = coordinate_dataframe.index.tolist()
                        
                        feature_dataframe = X_10[X_10.columns.difference(['index_orig','X','Y'])]
                        feature_list = feature_dataframe.values.tolist()
                        # 欧式距离 大于2.9就是0 不然就是1
                        coordinate_matrix = euclidean_distances(coordinate_list, coordinate_list)
                        coordinate_matrix = np.where(coordinate_matrix > 2.9, 0 , 1)
                        # 余弦相似度 阈值设置threshold 默认0.75
                        cosine_matrix = cosine_similarity(feature_list, feature_list)
                        
                        # 欧氏距离和余弦值均符合阈值的赋值为1
                        Adj_list = (coordinate_matrix == 1).astype(int) * (cosine_matrix >= threshold).astype(int)

                        # 将符合要求的node的index放入字典          
                        for c, item in enumerate(Adj_list):
                            for node_index in np.array(index_list)[item.astype('bool')]:
                                if node_index == index_list[c]:
                                    pass
                                else:
                                    node_dict[index_list[c]].append(node_index)
                                    
            
                        pbar.update()
            
            a_file = open(os.path.join(origin_dir, sample + '_node_dict.pkl'), "wb")
            pickle.dump(node_dict, a_file)
            a_file.close()
            dict_len_list = []
            
            for i in range(0, len(node_dict)):
                dict_len_list.append(len(node_dict[i]))
        
            arglist_strict = np.argsort(np.array(dict_len_list))
            arglist_strict = arglist_strict[::-1] # 输出从大到小的索引
            
            # 删除做对应点的索引 
            for arg_value in arglist_strict:
                if arg_value in node_dict.keys():
                    for adj_item in node_dict[arg_value]:
                        if adj_item in node_dict.keys():
                            # 删除元素
                            node_dict.pop(adj_item)
                            arglist_strict=np.delete(arglist_strict, np.argwhere(arglist_strict == adj_item))
                
            for key_value in node_dict.keys():
                node_dict[key_value] = list(set(node_dict[key_value]))
            

            #### 做超级节点 ###
            supernode_coordinate_x_strict = []
            supernode_coordinate_y_strict = []
            supernode_feature_strict = []
            
            supernode_relate_value = [supernode_coordinate_x_strict,
                                    supernode_coordinate_y_strict,
                                    supernode_feature_strict]
            
            whole_feature = graph_dataframe[graph_dataframe.columns.difference(['index_orig','X','Y'])]

            with tqdm(total = len(node_dict.keys())) as pbar_node:
                for key_value in node_dict.keys():
                    supernode_relate_value[0].append(graph_dataframe['X'][key_value])
                    supernode_relate_value[1].append(graph_dataframe['Y'][key_value])
                    if len(node_dict[key_value]) == 0:
                        # 无对应点的中心点，直接使用对应点的特征
                        select_feature = whole_feature.iloc[key_value]
                    else:
                        # 多的话就去平均值作为特征
                        select_feature = whole_feature.iloc[node_dict[key_value] + [key_value]]
                        select_feature = select_feature.mean()
                    # 放入supernode的特征文件中
                    if len(supernode_relate_value[2]) == 0:
                        temp_select = np.array(select_feature)
                        supernode_relate_value[2] = np.reshape(temp_select, (1,1794))
                    else:
                        temp_select = np.array(select_feature)
                        supernode_relate_value[2] = np.concatenate((supernode_relate_value[2], np.reshape(temp_select, (1,1794))), axis=0)
                    pbar_node.update()

            # 超级节点的空间距离，阈值设置为5.5,后续的superpath_network函数参数可调节超级节点距离生成不同图
            coordinate_integrate = pd.DataFrame({'X':supernode_relate_value[0],'Y':supernode_relate_value[1]})
            coordinate_matrix1 = euclidean_distances(coordinate_integrate, coordinate_integrate)
            coordinate_matrix1 = np.where(coordinate_matrix1 > spatial_threshold , 0 , 1)
            
            # 创建节点图
            fromlist = []
            tolist = []
            # 为啥不直接？
            # fromlist = np.unique(Edge_label[0], return_counts=True)
            # tolist = np.unique(Edge_label[1], return_counts=True)

            with tqdm(total = len(coordinate_matrix1)) as pbar_pytorch_geom:
                for i in range(len(coordinate_matrix1)):
                    temp = coordinate_matrix1[i,:]
                    selectindex = np.where(temp > 0)[0].tolist()
                    for index in selectindex:
                        fromlist.append(int(i))
                        tolist.append(int(index))
                    pbar_pytorch_geom.update()

            edge_index = torch.tensor([fromlist, tolist], dtype=torch.long)
            x = torch.tensor(supernode_relate_value[2], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index)

            node_dict = pd.DataFrame.from_dict(node_dict, orient='index')
            node_dict.to_csv(os.path.join(superpatch_dir, sample + '_' + str(threshold) + '.csv'))
            torch.save(data, os.path.join(superpatch_dir, sample+ '_' + str(threshold) + '_graph_torch.pt'))
    else:
        print('个数：{},没有＞2'.format(len(downsampling)))

def Parser_main():
    
    parser = argparse.ArgumentParser(description="TEA-graph superpatch generation")
    parser.add_argument("--database", default='GC', help="Use in the savedir", type = str)
    parser.add_argument("--cancertype",default='GC',help="cancer type",type=str)
    parser.add_argument("--graphdir",default="<path_save_graph>",help="graph save dir",type=str)
    parser.add_argument("--imagedir",default="<svs_file>",help="svs file location",type=str)
    parser.add_argument("--weight_path",default=None,help="pretrained weight path",type=str)
    parser.add_argument("--imagesize", default = 224, help ="crop image size", type = int)
    parser.add_argument("--threshold", default = 0.75, help = "cosine similarity threshold", type = float)
    parser.add_argument("--spatial_threshold", default = 5.5, help = "spatial threshold", type = float)
    parser.add_argument("--gpu", default = '0' , help = "gpu device number", type = str)
    return parser.parse_args()

def main():
    
    Argument = Parser_main()
    cancer_type = Argument.cancertype
    database = Argument.database
    image_dir = Argument.imagedir
    save_dir = Argument.graphdir
    gpu = Argument.gpu
    files = os.listdir(image_dir)
    
    # mata_data = pd.read_csv('GC_2300_all.csv', sep='\t')
    # mata_data = pd.read_csv('GC_2300_all.csv')
    # csv_id = list(mata_data['case_submitter_id'])
    # ids = []
    # for id in files:
    #     if id[:12] in csv_id:
    #         ids.append(id)
    # files = [file.replace("png", "svs")for file in ids]
    # ipdb.set_trace()



    save_dir = os.path.join(save_dir, database,cancer_type)

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)



    final_files = [os.path.join(image_dir, file) for file in files]

    final_files.sort(key=lambda f: os.stat(f).st_size, reverse=False)
    
    device = torch.device(int(gpu) if torch.cuda.is_available() else "cpu")
    model_ft = EfficientNet.from_pretrained('efficientnet-b4', num_classes = 2)
    if Argument.weight_path is not None:
        weight_path = Argument.weight_path
        load_weight = torch.load(weight_path, map_location = device)
        model_ft.load_state_dict(load_weight)
    model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)
    model_ft.eval()


    clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
    with open("/home/stat-huamenglei/SISH-main/checkpoints/trash_lgrlbp.pkl", 'rb') as handle:
        clf = pickle.load(handle)
    # ipdb.set_trace()
    for image in tqdm(final_files):
        id = image.split('/')[-1][:-4]+'_0.75_graph_torch.pt'
        # if image.split('/')[-1][:-4] =='TCGA-CD-8525':
        #     ipdb.set_trace()
        path = os.path.join(save_dir,'superpatch',id)
        if os.path.exists(path):
            print('{}存在，就跳过'.format(path))
            continue
        else:
            supernode_generation(image, model_ft,clf, device, Argument, save_dir)
    ipdb.set_trace()
    # 不同的空间阈值
    superpatch_network_construction.false_graph_filtering(4.3)
    
if __name__ == "__main__":
    main()