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
from skimage.filters import threshold_multiotsu
import pickle
import argparse
import ipdb
from glob import glob
import staintools
from PIL import Image
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import multiprocessing as mp
import sys



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



def process_svs(row):
    size = 256
    vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=0, getH_mode=0, ITER=50)
    model = KMeans(n_clusters=2, random_state=0, n_init=10)
    slideimage = osd.OpenSlide(row[1]['slide_path'])
    x_y_list = []
    Ws = []
    Hs = []
    Center = []
    _,x,y,_ = row[1]
    filter_location = (x*size, y*size)
    
    level = 0
    patch_size = (size, size)
    location = (filter_location[0], filter_location[1])
    CutImage = slideimage.read_region(location, level, patch_size)
    image = CutImage.convert('RGB')
    img = np.array(image)
    p = np.percentile(img, 90)
    img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)
    ws, hs = vhd.stain_separate(img)
    model.fit(hs.T)
    center = model.cluster_centers_

    return Ws.reshape(-1),Hs.reshape(-1),Center.reshape(-1),x_y_list

if __name__ == "__main__":
    WSI_path = sys.argv[1:][0]
    id = WSI_path.split('/')[-1][:-4]
    # ipdb.set_trace()
    print("============================Processing {} now==============================".format(WSI_path))
    csv_path  = '/mnt/usb/huamenglei/GC_2300/GC/original/{}_node_location_list.csv'.format(id)
    patch_save_path = '/mnt/usb/huamenglei/GC_2300/GC/original/{}'.format(id)
    all_tissue_samples = pd.read_csv(csv_path)
    all_tissue_samples['slide_path'] = WSI_path



    # ipdb.set_trace()

        
    with mp.Pool(processes=24) as pool:
        # Map the function to the rows of the dataframe
        results = pool.map(process_svs, [row  for row in all_tissue_samples.iterrows()])
    ipdb.set_trace()

    with open('results_list.pkl', 'wb') as f:
        pickle.dump(results, f)
