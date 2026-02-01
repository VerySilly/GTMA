# -*- coding: utf-8 -*-
import os
import copy
import torch
import torch_geometric.transforms as T
import pandas as pd
import numpy as np
import logging

from torch import optim
from torch_geometric.transforms import Polar
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, OneCycleLR

from tqdm import *

from model_selection import model_selection
from utils import train_test_split
from utils import makecheckpoint_dir_graph as mcd
from utils import TrainValid_path
from utils import non_decay_filter
from utils import coxph_loss
from utils import cox_sort
from utils import accuracytest

from torch.utils.data.sampler import Sampler

import ipdb

class Sampler_custom(Sampler):

    def __init__(self, event_list, censor_list, batch_size):
        self.event_list = event_list
        self.censor_list = censor_list
        self.batch_size = batch_size

    def __iter__(self):
        # Event_idx 是死亡
        # Censored_idx 是删失
        train_batch_sampler = []
        Event_idx = copy.deepcopy(self.event_list)
        Censored_idx = copy.deepcopy(self.censor_list)
        np.random.shuffle(Event_idx)
        np.random.shuffle(Censored_idx)

        # 确保死亡事件个数可以被2整除
        Int_event_batch_num = Event_idx.shape[0] // 2
        Int_event_batch_num = Int_event_batch_num * 2
        Event_idx_batch_select = np.random.choice(Event_idx.shape[0], Int_event_batch_num, replace=False)
        Event_idx = Event_idx[Event_idx_batch_select]

        # 确保删失事件个数可以被（batch_size - 2）整除
        Int_censor_batch_num = Censored_idx.shape[0] // (self.batch_size - 2)
        Int_censor_batch_num = Int_censor_batch_num * (self.batch_size - 2)
        Censored_idx_batch_select = np.random.choice(Censored_idx.shape[0], Int_censor_batch_num, replace=False)
        Censored_idx = Censored_idx[Censored_idx_batch_select]

        # 做成二维数组的形式
        Event_idx_selected = np.random.choice(Event_idx, size=(len(Event_idx) // 2, 2), replace=False)
        Censored_idx_selected = np.random.choice(Censored_idx, 
                                                 size=((Censored_idx.shape[0] // (self.batch_size - 2)), (self.batch_size - 2)), 
                                                 replace=False)

        if Event_idx_selected.shape[0] > Censored_idx_selected.shape[0]:
            # 当死亡事件的二维数组的第一维度大于删失事件的第一维度
            # 死亡：(243, 2)  
            # 删失：(252, 4)
            Event_idx_selected = Event_idx_selected[:Censored_idx_selected.shape[0],:]
        else:
            Censored_idx_selected = Censored_idx_selected[:Event_idx_selected.shape[0],:]

        for c in range(Event_idx_selected.shape[0]):
            train_batch_sampler.append(
                Event_idx_selected[c, :].flatten().tolist() + 
                Censored_idx_selected[c, :].flatten().tolist())

        return iter(train_batch_sampler)

    def __len__(self):
        return len(self.event_list) // 2

class CoxGraphDataset(Dataset):

    def __init__(self, filelist, survlist, stagelist, censorlist, Metadata, mode, model, transform=None, pre_transform=None):
        super(CoxGraphDataset, self).__init__()
        self.filelist = filelist
        self.survlist = survlist
        self.stagelist = stagelist
        self.censorlist = censorlist
        self.Metadata = Metadata
        self.mode = mode
        self.model = model
        self.polar_transform = Polar()

    def processed_file_names(self):
        return self.filelist

    def len(self):
        return len(self.filelist)

    def get(self, idx):
        # print('成功')
        data_origin = torch.load(self.filelist[idx])
        transfer = T.ToSparseTensor()
        item = self.filelist[idx].split('/')[-1].split('.pt')[0].split('_')[0][:7]
        mets_class = 0

        survival = self.survlist[idx]
        phase = self.censorlist[idx]
        stage = self.stagelist[idx]
        
        # 1792? 1794?
        data_re = Data(x=data_origin.x[:,:1792], edge_index=data_origin.edge_index)

        mock_data = Data(x=data_origin.x[:,:1792], edge_index=data_origin.edge_index, pos=data_origin.pos)

        data_re.pos = data_origin.pos
        data_re_polar = self.polar_transform(mock_data)
        polar_edge_attr = data_re_polar.edge_attr

        if (data_re.edge_index.shape[1] != data_origin.edge_attr.shape[0]):
            print('error!')
            print(self.filelist[idx].split('/')[-1])
        else:
            data = transfer(data_re)
            data.survival = torch.tensor(survival)
            data.phase = torch.tensor(phase)
            data.mets_class = torch.tensor(mets_class)
            data.stage = torch.tensor(stage)
            data.item = item
            data.edge_attr = polar_edge_attr
            data.pos = data_origin.pos

        return data

def val(model,dataloader,optimizer_ft,Cox_loss,epoch):
    model.eval()
    grad_flag = False
    with torch.set_grad_enabled(grad_flag):
        loss = 0
        risk_loss = 0
        EpochSurv = []
        EpochPhase = []
        EpochRisk = []

        EpochID = []
        EpochStage = []
        Epochloss = 0

        batchcounter = 1
        pass_count = 0
        for c, d in enumerate(dataloader,1):

            optimizer_ft.zero_grad()

            tempsurvival = torch.tensor([data.survival for data in d])
            tempphase = torch.tensor([data.phase for data in d])
            tempID = np.asarray([data.item for data in d])
            tempstage = torch.tensor([data.stage for data in d])
            tempmeta = torch.tensor([data.mets_class for data in d])
            # ipdb.set_trace()
            out = model(d)
            # 根据输出的风险进行排序后再进行loss cindex计算
            risklist, tempsurvival, tempphase, tempmeta, EpochSurv, EpochPhase, EpochRisk, EpochStage = \
                cox_sort(out, tempsurvival, tempphase, tempmeta, tempstage, tempID,
                            EpochSurv, EpochPhase, EpochRisk, EpochStage, EpochID)
            
            # 确保每次进入的batch里面有两个sensor的数据，否则就跳过loss计算
            if torch.sum(tempphase).cpu().detach().item() < 1:
                pass_count += 1
            else:
                risk_loss = Cox_loss(risklist, tempsurvival, tempphase)
                # 奇怪!
                loss = risk_loss

                Epochloss += loss.cpu().detach().item()

                batchcounter += 1
                risklist = []
                tempsurvival = []
                tempphase = []
                tempstage = []
                final_updated_feature_list = []
                updated_feature_list = []


        Epochacc = accuracytest(torch.tensor(EpochSurv), torch.tensor(EpochRisk),
                                torch.tensor(EpochPhase))
        Epochloss = Epochloss / batchcounter




        # print()

        print(' epoch:' + str(epoch)+" mode:val")
        print(" loss:" + str(Epochloss) + " acc:" + str(Epochacc) + " pass count:" + str(pass_count))
        logging.info(' epoch:' + str(epoch)+" mode:val" )
        logging.info(" loss:" + str(Epochloss) + " acc:" + str(Epochacc) + " pass count:" + str(pass_count))
        

    return Epochacc,Epochloss,pass_count


def test(model,dataloader,optimizer_ft,Cox_loss,epoch):
    model.eval()
    grad_flag = False
    with torch.set_grad_enabled(grad_flag):
        loss = 0
        risk_loss = 0
        EpochSurv = []
        EpochPhase = []
        EpochRisk = []

        EpochID = []
        EpochStage = []
        Epochloss = 0

        batchcounter = 1
        pass_count = 0
        # ipdb.set_trace()
        # with tqdm(total=len(loader[mode])) as pbar:
        for c, d in enumerate(dataloader,1):
            # ipdb.set_trace()
            optimizer_ft.zero_grad()

            tempsurvival = torch.tensor([data.survival for data in d])
            tempphase = torch.tensor([data.phase for data in d])
            tempID = np.asarray([data.item for data in d])
            tempstage = torch.tensor([data.stage for data in d])
            tempmeta = torch.tensor([data.mets_class for data in d])
            # ipdb.set_trace()
            out = model(d)
            # 根据输出的风险进行排序后再进行loss cindex计算
            risklist, tempsurvival, tempphase, tempmeta, EpochSurv, EpochPhase, EpochRisk, EpochStage = \
                cox_sort(out, tempsurvival, tempphase, tempmeta, tempstage, tempID,
                            EpochSurv, EpochPhase, EpochRisk, EpochStage, EpochID)
            
            # 确保每次进入的batch里面有两个sensor的数据，否则就跳过loss计算
            if torch.sum(tempphase).cpu().detach().item() < 1:
                pass_count += 1
            else:
                risk_loss = Cox_loss(risklist, tempsurvival, tempphase)
                # 奇怪!
                loss = risk_loss

                Epochloss += loss.cpu().detach().item()


                batchcounter += 1
                risklist = []
                tempsurvival = []
                tempphase = []
                tempstage = []
                final_updated_feature_list = []
                updated_feature_list = []


        Epochacc = accuracytest(torch.tensor(EpochSurv), torch.tensor(EpochRisk),
                                torch.tensor(EpochPhase))
        Epochloss = Epochloss / batchcounter





        print(' epoch:' + str(epoch)+" mode:test")
        print(" loss:" + str(Epochloss) + " acc:" + str(Epochacc) + " pass count:" + str(pass_count))
        logging.info(' epoch:' + str(epoch)+" mode:test" )
        logging.info(" loss:" + str(Epochloss) + " acc:" + str(Epochacc) + " pass count:" + str(pass_count))
        

    return Epochacc,Epochloss,pass_count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, stop_epoch=100, patience=20, verbose=False):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            stop_epoch (int): Earliest epoch possible for stopping.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = 1000
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = 0
        self.stop_epoch = stop_epoch
        self.checkpoint = save_path
        self.best_epoch = 0 
    def __call__(self, val_acc,val_loss, model,epoch):
 
        # score = val_loss
    
        if epoch == 0:
            self.save_checkpoint(val_acc,val_loss, model,epoch)
        elif val_loss <= self.val_loss_min or val_acc >=self.val_acc_max:
            self.counter = 0
            self.best_epoch = epoch
            self.save_checkpoint(val_acc,val_loss, model,epoch)

        else:
            
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience or epoch > self.stop_epoch:
                self.early_stop = True
            
 
    def save_checkpoint(self,val_acc,val_loss, model,epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if val_loss < self.best_score :
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
                self.val_loss_min = val_loss
                self.best_score = self.val_loss_min
            elif val_acc > self.val_acc_max :
                print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
                self.val_acc_max = val_acc
        checkpointinfo = 'epoch-{},acc-{:4f},loss-{:4f}.pt'
        path = os.path.join(self.save_path, checkpointinfo.format(epoch, val_acc,val_loss))
        self.checkpoint = path
        torch.save(model.module.state_dict(), path)  # 单机多卡
        # torch.save(model.state_dict(), path)  # 单机单卡 这里会存储迄今最优模型的参数
 
        


def Train(Argument):

    checkpoint_dir, Figure_dir = mcd(Argument)
    # ipdb.set_trace()
    batch_num = int(Argument.batch_size) # 6个
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    Metadata = pd.read_csv('/GC_2300_all.csv')
    
    # TrainRoot = TrainValid_path(Argument.DatasetType)
    TrainRoot = '/mnt/usb/huamenglei/GC_2300/GC/superpatch'
    # pt 文件列表 
    Trainlist = os.listdir(TrainRoot)
    Trainlist = [item for c, item in enumerate(Trainlist) if '0.75_graph_torch_4.3_artifact_sophis_final.pt' in item]
    Fi = Argument.FF_number


    TrainFF_set, ValidFF_set, Test_set = train_test_split(Trainlist, Metadata, Argument.DatasetType, TrainRoot, Fi)


    TestDataset = CoxGraphDataset(filelist=Test_set[0], survlist=Test_set[1],
                                  stagelist=Test_set[3], censorlist=Test_set[2],
                                  Metadata=Metadata, mode=Argument.DatasetType,
                                  model=Argument.model)
    TrainDataset = CoxGraphDataset(filelist=TrainFF_set[0], survlist=TrainFF_set[1],
                                   stagelist=TrainFF_set[3], censorlist=TrainFF_set[2],
                                   Metadata=Metadata, mode=Argument.DatasetType,
                                   model=Argument.model)
    ValidDataset = CoxGraphDataset(filelist=ValidFF_set[0], survlist=ValidFF_set[1],
                                   stagelist=ValidFF_set[3], censorlist=ValidFF_set[2],
                                   Metadata=Metadata, mode=Argument.DatasetType,
                                   model=Argument.model)


    Event_idx = np.where(np.array(TrainFF_set[2]) == 1)[0]
    Censored_idx = np.where(np.array(TrainFF_set[2]) == 0)[0]
    train_batch_sampler = Sampler_custom(Event_idx, Censored_idx, batch_num)

    torch.manual_seed(12345)

    train_loader = DataListLoader(TrainDataset, batch_sampler=train_batch_sampler, num_workers=8, pin_memory=True)
    val_loader = DataListLoader(ValidDataset, batch_size=batch_num, shuffle=True, num_workers=8, pin_memory=True,
                            drop_last=False)


    model = model_selection(Argument)

    model_parameter_groups = non_decay_filter(model)

    model = DataParallel(model, device_ids=[0, 1], output_device=0)
    model = model.to(device)
    Cox_loss = coxph_loss()
    Cox_loss = Cox_loss.to(device)
    risklist = []
    optimizer_ft = optim.AdamW(model_parameter_groups, lr=Argument.learning_rate, weight_decay=Argument.weight_decay)
    scheduler = OneCycleLR(optimizer_ft, max_lr=Argument.learning_rate, steps_per_epoch=len(train_loader),
                           epochs=Argument.num_epochs)

    tempsurvival = []
    tempphase = []
    transfer = T.ToSparseTensor()
    bestloss = 100000
    bestacc = 0
    bestepoch = 0

    # loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    BestAccDict = {'train': 0, 'val': 0, }
    AccHistory = {'train': [], 'val': [], }
    LossHistory = {'train': [], 'val': [], }
    RiskAccHistory = {'train': [], 'val': [], 'test': []}
    RiskLossHistory = {'train': [], 'val': [], 'test': []}
    ClassAccHistory = {'train': [], 'val': [], 'test': []}
    ClassLossHistory = {'train': [], 'val': [], 'test': []}

    global_batch_counter = 0

    FFCV_accuracy = []
    FFCV_best_epoch = []
    logging.basicConfig(filename="{}/train_val_test.log".format(checkpoint_dir),level=logging.INFO)
    logging.info('Model:{},Loss_type:{},MLP_layernum:{}'.format(Argument.model,Argument.loss_type,Argument.MLP_layernum))
    count = 0
    early_stopping = EarlyStopping(save_path =checkpoint_dir, patience = 20, stop_epoch=60, verbose = True)
    for epoch in tqdm(range(0, int(Argument.num_epochs))):
        # phaselist = ['train', 'val', 'test']
        model.train()
        grad_flag = True
        with torch.set_grad_enabled(grad_flag):

            loss = 0
            risk_loss = 0
            class_loss = 0
            EpochSurv = []
            EpochPhase = []
            EpochRisk = []
            EpochTrueMeta = []
            EpochPredMeta = []
            EpochFeature = []
            EpochID = []
            EpochStage = []
            Epochloss = 0
            Aux_node_loss = 0
            Aux_edge_loss = 0
            Risk_loss = 0
            Epochriskloss = 0
            Epochclassloss = 0
            batchcounter = 1
            pass_count = 0
            # ipdb.set_trace()
            # with tqdm(total=len(loader[mode])) as pbar:
            for c, d in enumerate(train_loader, 1):
                # ipdb.set_trace()
                optimizer_ft.zero_grad()

                tempsurvival = torch.tensor([data.survival for data in d])
                tempphase = torch.tensor([data.phase for data in d])
                tempID = np.asarray([data.item for data in d])
                tempstage = torch.tensor([data.stage for data in d])
                tempmeta = torch.tensor([data.mets_class for data in d])
                ipdb.set_trace()
                out = model(d)
                # 根据输出的风险进行排序后再进行loss cindex计算
                risklist, tempsurvival, tempphase, tempmeta, EpochSurv, EpochPhase, EpochRisk, EpochStage = \
                    cox_sort(out, tempsurvival, tempphase, tempmeta, tempstage, tempID,
                                EpochSurv, EpochPhase, EpochRisk, EpochStage, EpochID)
                
                # 确保每次进入的batch里面有两个sensor的数据，否则就跳过loss计算
                if torch.sum(tempphase).cpu().detach().item() < 1:
                    pass_count += 1
                else:
                    risk_loss = Cox_loss(risklist, tempsurvival, tempphase)
                    # 奇怪!
                    loss = risk_loss
                    loss.backward()
                    # ipdb.set_trace()
                    torch.nn.utils.clip_grad_norm_(model_parameter_groups[0]['params'], max_norm=Argument.clip_grad_norm_value, 
                                                #    error_if_nonfinite=True
                                                    )
                    torch.nn.utils.clip_grad_norm_(model_parameter_groups[1]['params'], max_norm=Argument.clip_grad_norm_value, 
                                                #    error_if_nonfinite=True
                                                    )
                    optimizer_ft.step()
                    scheduler.step()

                    Epochloss += loss.cpu().detach().item()

                    batchcounter += 1
                    risklist = []
                    tempsurvival = []
                    tempphase = []
                    tempstage = []
                    final_updated_feature_list = []
                    updated_feature_list = []


            Epochacc = accuracytest(torch.tensor(EpochSurv), torch.tensor(EpochRisk),
                                    torch.tensor(EpochPhase))
            Epochloss = Epochloss / batchcounter


            # print()

            print(' epoch:' + str(epoch)+" mode:train")
            print(" loss:" + str(Epochloss) + " acc:" + str(Epochacc) + " pass count:" + str(pass_count))
            logging.info(' epoch:' + str(epoch)+" mode:train")
            logging.info(" loss:" + str(Epochloss) + " acc:" + str(Epochacc) + " pass count:" + str(pass_count))


            # checkpointinfo = 'epoch-{},acc-{:4f},loss-{:4f}.pt'

            
            # 验证集
            val_acc,val_loss,val_passcount = val(model,val_loader,optimizer_ft,Cox_loss,epoch)
            # test_acc,test_loss,test_passcount = test(model,test_loader,optimizer_ft,Cox_loss,epoch)
            # ipdb.set_trace()
            AccHistory['train'].append(Epochacc)
            AccHistory['val'].append(val_acc)
            # AccHistory['test'].append(test_acc)

            LossHistory['train'].append(Epochloss)
            LossHistory['val'].append(val_loss)
            # LossHistory['test'].append(test_acc)

            if Epochacc > BestAccDict['train']:
                BestAccDict['train'] = Epochacc


            if Epochacc > BestAccDict['val']:
                BestAccDict['val'] = Epochacc



            early_stopping(val_acc, val_loss, model, epoch)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break

    # ipdb.set_trace()
    AccHistory_df = pd.DataFrame.from_dict(AccHistory)
    AccHistory_df.to_csv(os.path.join(checkpoint_dir, 'AccHistory.csv'))

    LossHistory_df = pd.DataFrame.from_dict(LossHistory)
    LossHistory_df.to_csv(os.path.join(checkpoint_dir, 'LossHistory.csv'))

    # BestAccDict_df = pd.DataFrame.from_dict(BestAccDict)
    # BestAccDict_df.to_csv(os.path.join(checkpoint_dir, 'BestAccDict.csv'))

    # FFCV_accuracy.append(bestacc)
    # FFCV_best_epoch.append(bestepoch)

    # bestFi = np.argmax(FFCV_accuracy)
    # best_checkpoint_dir = os.path.join(checkpoint_dir, str(bestFi))
    # best_figure_dir = os.path.join(checkpoint_dir, str(bestFi))

    # Argument.checkpoint_dir = best_checkpoint_dir
    # logging.info(Argument.checkpoint_dir+str(FFCV_best_epoch[bestFi]))
    # return model, best_checkpoint_dir, best_figure_dir, FFCV_best_epoch[bestFi]
    return model,early_stopping.checkpoint,early_stopping.best_epoch