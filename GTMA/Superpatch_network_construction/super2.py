import pandas as pd
import numpy as np
import os
import openslide as osd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
import time
import ipdb
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch
import ResNet as ResNet
from torch.nn import nn
from tqdm import tqdm


class PatchesImageDatasetLoader(Dataset):
    def __init__(self,slideimage_path,locations, level_0_magnification,transform):
        self.slideimage = osd.OpenSlide(slideimage_path)
        self.locations = locations
        self.level_0_magnification = level_0_magnification
        self.transform = transform

    def __len__(self):
        return len(self.locations)
    

    def get_files(self, path, cls):


        return all

    def __getitem__(self, idx):

        i = self.locations.iloc[idx]['X']
        j = self.locations.iloc[idx]['Y']
        coor = (i,j)
        if self.level_0_magnification == 20:
            imagesize = 256
            filter_location = (i * imagesize, j * imagesize)
            level = 0
            patch_size = (imagesize, imagesize)
            location = (filter_location[0], filter_location[1])

            CutImage = self.slideimage.read_region(location, level, patch_size)
            img = CutImage.convert('RGB')
            if self.transform:
                img = self.transform(img)

        
        return img,coor


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)


def main():
    image_dir = '/mnt/usb3/1.WSI_数据/1.GC_2300/WSI数据'
    mata_path = '/home/stat-huamenglei/TEA-graph-master/GC_2300_all.csv'
    save_dir = '/mnt/usb/huamenglei/GC_2300/GC'

    files = os.listdir(image_dir)
    mata_data = pd.read_csv(mata_path)
    csv_id = list(mata_data['WSIs'])
    ids = [id for id in files if id[:12] in csv_id]
    final_files = [os.path.join(image_dir, file) for file in files]
    final_files.sort(key=lambda f: os.stat(f).st_size, reverse=False)
    delete_id = []
    count = 0

    model = ResNet.resnet50(num_classes=128,mlp=False, two_branch=False, normlinear=True)
    pretext_model = torch.load(r'/home/stat-huamenglei/RetCCL-main/best_ckpt.pth')
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)
    model.eval()
    device = torch.device(0 if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)
    for path in final_files:
        id = path.split('/')[-1][:-4]
        location_path = f'/mnt/usb/huamenglei/GC_2300/GC/original/{id}_node_location_list.csv'
        img_dir = f'/mnt/usb4/hml/GC_retccl_feature/{id}'
        supernode_path = f'/mnt/usb/huamenglei/GC_2300/GC/superpatch/{id}_0.75.csv'
        if not os.path.exists(img_dir): os.mkdir(img_dir)


        locations = pd.read_csv(location_path)
        slideimage_path = path


        
        slideimage = osd.OpenSlide(slideimage_path)
        downsampling = slideimage.level_downsamples



        if 'aperio.AppMag' in slideimage.properties.keys():
            level_0_magnification = int(slideimage.properties['aperio.AppMag'])
        elif 'openslide.mpp-x' in slideimage.properties.keys():
            level_0_magnification = 40 if int(np.floor(float(slideimage.properties['openslide.mpp-x']) * 10)) == 2 else 20
        else:
            level_0_magnification = '有问题'
        print('{}进入函数,downsampling:{},最大放大倍数{}'.format(id, downsampling, level_0_magnification))


        Dataset = PatchesImageDatasetLoader(slideimage_path,locations,level_0_magnification,trnsfrms_val)
        dataloader = torch.utils.data.DataLoader(Dataset,batch_size=16,num_workers=4,drop_last=False)



        inside_counter = 0
        feature_list = []
        x_y_list = []
        temp_x = []
        temp_y = []
        
        with torch.no_grad():
            for batch in dataloader:
                images ,coor = batch
                images = images.to(device)
                features = model(batch)
                # features = features.cpu().numpy()



                # 生成feature_list，x_y_list 特征和坐标的list
                if inside_counter == 0:
                    feature_list = np.concatenate((features.cpu().detach().numpy()), axis=1)
                    temp_x = np.reshape(np.array(temp_x), (len(temp_x),1))
                    temp_y = np.reshape(np.array(temp_y), (len(temp_x),1))
                    
                    x_y_list = np.concatenate((temp_x,temp_y),axis=1)
                else:
                    feature_list = np.concatenate((feature_list,np.concatenate((features.cpu().detach().numpy()),axis=1)), axis=0)
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


        feature_df = pd.DataFrame.from_dict(feature_list)
        coordinate_df = pd.DataFrame({'X': x_y_list[:,0],'Y': x_y_list[:,1]})
        graph_dataframe = pd.concat([coordinate_df, feature_df], axis = 1)
        graph_dataframe = graph_dataframe.sort_values(by = ['Y', 'X'])
        graph_dataframe = graph_dataframe.reset_index(drop = True)
        coordinate_df = graph_dataframe.iloc[:,0:2]
        feature_df.to_csv(os.path.join(img_dir, 'feature_list.csv'))
        coordinate_df.to_csv(os.path.join(img_dir, 'node_location_list.csv'))
        index = list(graph_dataframe.index)
        graph_dataframe.insert(0,'index_orig', index)
        
        node_dict = {}
        
        for i in range(len(coordinate_df)):
            node_dict.setdefault(i,[])
        
        X = max(set(np.squeeze(graph_dataframe.loc[:, ['X']].values,axis = 1)))
        Y = max(set(np.squeeze(graph_dataframe.loc[:, ['Y']].values, axis = 1)))


        
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
if __name__ == "__main__":
    main()
