# -*- coding: utf-8 -*-

import os
import sys
import argparse
from train import Train
from analyze import Analyze
import torch
import ipdb
from model_selection import model_selection
from torch_geometric.nn import DataParallel
from collections import OrderedDict
def Parser_main():
    parser = argparse.ArgumentParser(description="Deep cox analysis model")
    parser.add_argument("--DatasetType", default="GC", help="TCGA or GC",
                        type=str)# 默认是TCGA
    parser.add_argument("--learning_rate", default=0.0001, help="Learning rate", type=float)
    parser.add_argument("--weight_decay", default=0.00005, help="Weight decay rate", type=float)
    parser.add_argument("--clip_grad_norm_value", default=2.0, help="Gradient clipping value", type=float)
    parser.add_argument("--batch_size", default=6, help="batch size", type=int)
    parser.add_argument("--num_epochs", default=100, help="Number of epochs", type=int)
    parser.add_argument("--dropedge_rate", default=0.25, help="Dropedge rate for GAT", type=float)
    parser.add_argument("--dropout_rate", default=0.25, help="Dropout rate for MLP", type=float)
    parser.add_argument("--graph_dropout_rate", default=0.25, help="Node/Edge feature dropout rate", type=float)
    parser.add_argument("--initial_dim", default=100, help="Initial dimension for the GAT", type=int)
    parser.add_argument("--attention_head_num", default=2, help="Number of attention heads for GAT", type=int)
    parser.add_argument("--number_of_layers", default=3, help="Whole number of layer of GAT", type=int)
    parser.add_argument("--FF_number", default=0, help="Selecting set for the five fold cross validation", type=int)
    parser.add_argument("--model", default="GAT_custom", help="GAT_custom/DeepGraphConv/PatchGCN/GIN/MIL/MIL-attention", type=str)
    parser.add_argument("--gpu", default=0, help="Target gpu for calculating loss value", type=int)
    parser.add_argument("--norm_type", default="layer", help="BatchNorm=batch/LayerNorm=layer", type=str)
    parser.add_argument("--MLP_layernum", default=3, help="Number of layers for pre/pose-MLP", type=int)
    parser.add_argument("--with_distance", default="Y", help="Y/N; Including positional information as edge feature", type=str)
    parser.add_argument("--simple_distance", default="N", help="Y/N; Whether multiplying or embedding positional information", type=str)
    parser.add_argument("--loss_type", default="PRELU", help="RELU/Leaky/PRELU", type=str)
    parser.add_argument("--residual_connection", default="Y", help="Y/N", type=str)

    return parser.parse_args()

def main():
    Argument = Parser_main()
    best_model,checkpoint_dir,bestepoch = Train(Argument)
    print('bestepoch: {} checkpoint_dir: {}'.format(bestepoch,checkpoint_dir))    
    torch.save(best_model.state_dict(), os.path.join(checkpoint_dir))
    


    # # 测试最佳模型
    checkpoint_dir = 'GTMA/results/GC_5/GAT_custom/0FOLD/0'
    fig_dir = 'GTMA/results/GC_5/GAT_custom/0FOLD/0'
    bestepoch = 45
    model = model_selection(Argument)

    
    state_dict = torch.load('GTMA/results/GC_5/GAT_custom/0FOLD/epoch-45,acc-0.738182,loss-1.366421.pt')
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v 
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(state_dict)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = DataParallel(model, device_ids=[0, 1], output_device=0)
    model = model.to(device)
    best_model = model
    Analyze(Argument, best_model, checkpoint_dir, fig_dir, bestepoch, best_select="Y")

if __name__ == "__main__":
    main()
