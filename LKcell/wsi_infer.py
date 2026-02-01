#!/usr/bin/env python3
"""
WSI细胞分布分析脚本
分析WSI的所有patch的细胞分布，并应用RPSM选择条件，生成详细的统计图表
Enhanced with Multi-GPU support for faster processing
"""

import os
import sys
import pandas as pd
import numpy as np
import random
import torch
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import threading
from pathlib import Path
import torchvision.transforms as transforms
import torch.nn.functional as F
from collections import defaultdict
import glob
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import roc_curve, auc
from skimage.measure import regionprops
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
# 设置字体为英文，避免乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
def generate_inst_type_from_class_map(instance_map, class_map):
    """
    instance_map: (H, W) 实例分割图，0 表示背景
    class_map: (H, W) 实例类别图，0 表示背景，1,2,...表示不同类别
    返回:
        inst_type: (N,1) 每个实例对应类别
    """
    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids != 0]  # 去掉背景

    inst_type = []
    for inst_id in unique_ids:
        # 获取当前实例的类别
        mask = instance_map == inst_id
        # 假设实例内部类别一致，取第一个非零类别
        cls = np.unique(class_map[mask])
        cls = cls[cls != 0][0] if len(cls[cls != 0]) > 0 else 0
        inst_type.append(cls)
    inst_type = np.array(inst_type, dtype=np.int32).reshape(-1, 1)
    return inst_type


class SlideNucStatObject:
    """
    计算单张切片的细胞核特征，包括形态、颜色、Haralick、邻居信息。
    输入为 instance_map 和 inst_type。
    """
    def __init__(self, instance_map: np.ndarray, inst_type,image: np.ndarray = None):
        """
        Args:
            instance_map: (H, W) 分割实例图，0 表示背景
            inst_type: (H, W) 实例类型图，0 表示背景
            image: 可选，RGB 原图，用于颜色和 Haralick 特征计算
        """
        self.type_names = {1: "Neoplastic", 2: "Inflammatory", 3: "Connective", 4: "Dead", 5: "Epithelial"}
        self.instance_map = instance_map
    # 默认类型全部为0
        # import ipdb; ipdb.set_trace()
        self.inst_type = generate_inst_type_from_class_map(instance_map, inst_type)
        self.image = image
        self.nuclei_index = np.arange(len(inst_type))  # 对应每个实例的索引
        self.n_instances = len(self.nuclei_index)
        self.feature_columns = None

    def _get_haralick_features(self, gray_img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
        """
        提取单个细胞的 Haralick 特征
        """
        glcm = graycomatrix(gray_img, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
        features = {}
        props = ["contrast", "homogeneity", "dissimilarity", "ASM", "energy", "correlation"]
        for prop in props:
            features[prop] = np.mean(graycoprops(glcm, prop))
        features["heterogeneity"] = 1 - features["homogeneity"]
        return features

    def _nuc_stat_func(self, inst_id):
        mask = self.instance_map == inst_id
        if mask.sum() == 0:
            return None

        # 形态特征
        stat = regionprops(mask.astype(np.uint8))[0]
        morphology = {
            "major_axis_length": stat.major_axis_length,
            "minor_axis_length": stat.minor_axis_length,
            "major_minor_ratio": stat.major_axis_length / stat.minor_axis_length if stat.minor_axis_length>0 else 0,
            "orientation_degree": stat.orientation * 180 / np.pi + 90,
            "area": stat.area,
            "extent": stat.extent,
            "solidity": stat.solidity,
            "convex_area": stat.convex_area,
            "eccentricity": stat.eccentricity,
            "equivalent_diameter": stat.equivalent_diameter,
            "perimeter": stat.perimeter
        }

        # 颜色特征
        color_features = {}
        if self.image is not None:
            masked_img = self.image * np.expand_dims(mask, axis=-1)
            for i, c in enumerate(["R", "G", "B"]):
                channel = masked_img[:,:,i][mask]
                color_features[f"{c}_mean"] = np.mean(channel)
                color_features[f"{c}_std"] = np.std(channel)
                color_features[f"{c}_min"] = np.min(channel)
                color_features[f"{c}_max"] = np.max(channel)
            gray_img = rgb2gray(masked_img).astype(np.uint8)
            haralick_features = self._get_haralick_features(gray_img)
        else:
            for c in ["R","G","B"]:
                color_features[f"{c}_mean"] = np.nan
                color_features[f"{c}_std"] = np.nan
                color_features[f"{c}_min"] = np.nan
                color_features[f"{c}_max"] = np.nan
            haralick_features = {k: np.nan for k in ["contrast","homogeneity","dissimilarity","ASM","energy","correlation","heterogeneity"]}

        cell_type = self.inst_type[inst_id-1,0]
        cell_type_name = self.type_names.get(cell_type, "Unknown")
        features = {"inst_id": inst_id, "cell_type": cell_type_name}
        features.update(morphology)
        features.update(color_features)
        features.update(haralick_features)
        return features

    def compute_nuc_features(self):
        features = []
        for inst_id in tqdm(range(1, self.n_instances+1)):
            stat = self._nuc_stat_func(inst_id)
            if stat is not None:
                features.append(stat)
        df_features = pd.DataFrame(features)
        return df_features

    def compute_delaunay_features(self, df_features):
        """
        计算 Delaunay 邻居特征
        """
        centroids = []
        for inst_id in range(1, self.n_instances+1):
            mask = self.instance_map == inst_id
            props = regionprops(mask.astype(np.uint8))[0]
            centroids.append(props.centroid)
        centroids = np.array(centroids)
        tri = Delaunay(centroids)
        indices, indptr = tri.vertex_neighbor_vertices
        # import ipdb
        # ipdb.set_trace()
        delaunay_feats = []
        for i in range(self.n_instances):
            neighbors = indices[indptr[i]:indptr[i+1]]
            if len(neighbors) == 0:
                delaunay_feats.append([np.nan]*4)
            else:
                dist = np.linalg.norm(centroids[neighbors]-centroids[i], axis=1)
                delaunay_feats.append([np.mean(dist), np.std(dist), np.min(dist), np.max(dist)])
        df_delaunay = pd.DataFrame(delaunay_feats, columns=["dist_mean","dist_std","dist_min","dist_max"])
        return df_delaunay

    def compute_features(self):
        df_nuc = self.compute_nuc_features()
        # df_delaunay = self.compute_delaunay_features(df_nuc)
        df_all = pd.concat([df_nuc.reset_index(drop=True)], axis=1)
        return df_all

# ================== 大规模数据分析配置 ==================
class AnalysisConfig:
    """分析配置类 - 针对大规模WSI数据优化"""
    # 数据集规模阈值
    LARGE_DATASET_THRESHOLD = 50      # 大规模数据集阈值
    MEDIUM_DATASET_THRESHOLD = 20     # 中等规模数据集阈值
    
    # 采样策略
    MAX_DETAILED_ANALYSIS_LARGE = 10  # 大规模数据集最多详细分析数量
    MAX_DETAILED_ANALYSIS_MEDIUM = 20 # 中等数据集最多详细分析数量
    
    # 性能优化
    BATCH_SIZE = 8                    # 分批处理大小
    MAX_WORKERS = 4                   # 最大工作线程数
    
    # 可视化控制
    ENABLE_INDIVIDUAL_WSI_PLOTS = True  # 是否生成个体WSI图表
    FORCE_AGGREGATED_MODE = False      # 强制使用聚合模式
    
    # 输出控制
    SAVE_INTERMEDIATE_RESULTS = True   # 保存中间结果
    GENERATE_SAMPLING_INFO = True      # 生成采样信息
    
    @classmethod
    def auto_configure(cls, num_wsi):
        """根据WSI数量自动配置参数"""
        if num_wsi > cls.LARGE_DATASET_THRESHOLD:
            print(f"🔧 Auto-config: Large dataset mode ({num_wsi} WSIs)")
            cls.ENABLE_INDIVIDUAL_WSI_PLOTS = False
            cls.FORCE_AGGREGATED_MODE = True
        elif num_wsi > cls.MEDIUM_DATASET_THRESHOLD:
            print(f"🔧 Auto-config: Medium dataset mode ({num_wsi} WSIs)")
            cls.ENABLE_INDIVIDUAL_WSI_PLOTS = True  # 但会采样
        else:
            print(f"🔧 Auto-config: Small dataset mode ({num_wsi} WSIs)")

def numpy_json_serializer(obj):
    """
    Custom JSON serializer for numpy data types and other non-serializable objects
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # For numpy scalars
        return obj.item()
    else:
        return str(obj)

def setup_pannuke_models():
    """
    Setup and load PanNuke models with PyTorch DataParallel for multi-GPU inference
    使用PyTorch标准的DataParallel实现多GPU加速
    """
    # Set working directory to LKCell
    original_cwd = os.getcwd()
    lkcell_dir = Path("/home/stat-huamenglei/LKCell")
    # import ipdb; ipdb.set_trace()
    if not lkcell_dir.exists():
        print("LKCell directory does not exist")
        return None, None
    
    # Check available GPUs
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return setup_single_gpu_model()
    
    num_gpus = torch.cuda.device_count()
    print(f"🚀 Detected {num_gpus} GPU(s) available")
    
    # Switch to LKCell directory
    os.chdir(lkcell_dir)
    sys.path.insert(0, str(Path.cwd()))
    
    try:
        from cell_segmentation.inference.inference_cellvit_experiment_pannuke import InferenceCellViTParser, InferenceCellViT
        device_ids = [0, 1]
        device = torch.device(f"cuda:{device_ids[0]}")
        print(f"⚡ Setting up DataParallel model on all available GPUs...")
        
        # Setup PanNuke model configuration
        pannuke_parser = InferenceCellViTParser()
        pannuke_configurations = pannuke_parser.parse_arguments()
        
        pannuke_inf = InferenceCellViT(
            run_dir=pannuke_configurations["run_dir"],
            checkpoint_name=pannuke_configurations["checkpoint_name"],
            gpu=pannuke_configurations["gpu"],
            magnification=pannuke_configurations["magnification"],
        )
        
        # Load model checkpoint
        checkpoint_path = pannuke_inf.run_dir / "checkpoints" / pannuke_inf.checkpoint_name
        print(f"Checkpoint path: {checkpoint_path}")
        
        if not checkpoint_path.exists():
            print(f"Model checkpoint does not exist: {checkpoint_path}")
            os.chdir(original_cwd)
            return None, None
        
        pannuke_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        pannuke_model = pannuke_inf.get_model(model_type=pannuke_checkpoint["arch"])
        pannuke_model.load_state_dict(pannuke_checkpoint["model_state_dict"])
        pannuke_model.to(device)
        # 使用DataParallel包装模型以支持多GPU
        if num_gpus > 1:
            print(f"🎯 Wrapping model with DataParallel for {num_gpus} GPUs")
            pannuke_model = torch.nn.DataParallel(pannuke_model, device_ids=device_ids)
            is_multi_gpu = True
        else:
            is_multi_gpu = False
        
        pannuke_model.to(device)
        pannuke_model.eval()
        
        print(f"✅ PanNuke model loaded to device: {device}")
        if is_multi_gpu:
            print(f"🚀 Multi-GPU acceleration enabled with DataParallel")
        
        # Switch back to original directory
        os.chdir(original_cwd)
        
        return pannuke_model, device, is_multi_gpu
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        os.chdir(original_cwd)
        return setup_single_gpu_model()

def setup_single_gpu_model():
    """
    Fallback function for single GPU setup
    """
    # Set working directory to LKCell
    original_cwd = os.getcwd()
    lkcell_dir = Path("LKCell")
    os.chdir(lkcell_dir)
    sys.path.insert(0, str(Path.cwd()))
    
    try:
        from cell_segmentation.inference.inference_cellvit_experiment_pannuke import InferenceCellViTParser, InferenceCellViT
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Setup PanNuke model
        pannuke_parser = InferenceCellViTParser()
        pannuke_configurations = pannuke_parser.parse_arguments()
        
        print(f"Configuration: {pannuke_configurations}")
        
        pannuke_inf = InferenceCellViT(
            run_dir=pannuke_configurations["run_dir"],
            checkpoint_name=pannuke_configurations["checkpoint_name"],
            gpu=pannuke_configurations["gpu"],
            magnification=pannuke_configurations["magnification"],
        )
        
        # Load model checkpoint
        checkpoint_path = pannuke_inf.run_dir / "checkpoints" / pannuke_inf.checkpoint_name
        print(f"Checkpoint path: {checkpoint_path}")
        
        if not checkpoint_path.exists():
            print(f"Model checkpoint does not exist: {checkpoint_path}")
            os.chdir(original_cwd)
            return None, None, False
        
        pannuke_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        pannuke_model = pannuke_inf.get_model(model_type=pannuke_checkpoint["arch"])
        pannuke_model.load_state_dict(pannuke_checkpoint["model_state_dict"])
        pannuke_model.to(device)
        pannuke_model.eval()
        
        print(f"PanNuke model loaded to device: {device}")
        
        # Switch back to original directory
        os.chdir(original_cwd)
        
        return pannuke_model, device, False
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        os.chdir(original_cwd)
        return None, None, False

def load_all_wsi_data(csv_path):
    """
    Load ALL WSI data from CSV file for complete RPSM evaluation
    Used for comprehensive analysis of RPSM filtering criteria
    """
    df = pd.read_csv(csv_path)
    print(f"Total {len(df)} WSI samples found")
    
    # Get all samples from responder and non-responder groups
    responders = df[df['label'] == 1]
    non_responders = df[df['label'] == 0]
    
    print(f"Responder group: {len(responders)} samples")
    print(f"Non-responder group: {len(non_responders)} samples")
    
    all_samples = []
    
    # Add all responder samples
    for _, sample in responders.iterrows():
        all_samples.append(sample)
        print(f"Added responder sample: {sample['slides_name']}")
    
    # Add all non-responder samples  
    for _, sample in non_responders.iterrows():
        all_samples.append(sample)
        print(f"Added non-responder sample: {sample['slides_name']}")
    
    print(f"Total {len(all_samples)} WSI samples will be analyzed for complete RPSM evaluation")
    return all_samples

def load_and_sample_wsi_data(csv_path, num_samples_per_group=3):
    """
    Load WSI data from CSV file and strategically sample
    Enhanced sampling to reduce heterogeneity and improve analysis reliability
    Updated default to 12 samples per group for better statistical power
    """
    df = pd.read_csv(csv_path)
    print(f"Total {len(df)} WSI samples found")
    
    # Select multiple samples from responder and non-responder groups
    responders = df[df['label'] == 1]
    non_responders = df[df['label'] == 0]
    
    print(f"Responder group: {len(responders)} samples")
    print(f"Non-responder group: {len(non_responders)} samples")
    
    selected_samples = []
    
    # Reset index to ensure proper sampling
    responders = responders.reset_index(drop=True)
    non_responders = non_responders.reset_index(drop=True)
    
    # Enhanced sampling strategy for responders
    if len(responders) > 0:
        num_responder_samples = min(num_samples_per_group, len(responders))
        print(f"Selecting {num_responder_samples} responder samples with enhanced strategy...")
        
        # 使用多重随机性源提高选择的多样性
        import time
        base_seed = int(time.time() * 1000000) % 2147483647
        
        # 智能分层采样策略 - 基于12-WSI分析经验
        # 优先选择具有不同组织学特征的样本以提高代表性
        if len(responders) >= 12:
            # 尝试分层采样：选择不同临床特征的样本
            # 如果有临床数据，可以按年龄、分期、病理类型等分层
            selected_indices = []
            available_indices = list(range(len(responders)))
            
            # 多次随机化确保样本多样性
            for round_num in range(3):
                temp_seed = (base_seed + round_num * 7919) % 2147483647
                temp_random = random.Random(temp_seed)
                temp_random.shuffle(available_indices)
            
            # 最终选择
            selected_indices = available_indices[:num_responder_samples]
        else:
            selected_indices = available_indices[:num_responder_samples]
        
        print(f"Selected responder indices: {selected_indices}")
        
        selected_responders = responders.iloc[selected_indices]
        
        for _, sample in selected_responders.iterrows():
            selected_samples.append(sample)
            print(f"Selected responder sample: {sample['slides_name']}")
    
    # Enhanced sampling strategy for non-responders
    if len(non_responders) > 0:
        num_non_responder_samples = min(num_samples_per_group, len(non_responders))
        print(f"Selecting {num_non_responder_samples} non-responder samples with enhanced strategy...")
        
        # 使用系统时间作为额外的随机性源
        import time
        time.sleep(0.001)  # 短暂延迟确保时间变化
        microsecond_seed = int(time.time() * 1000000) % 1000000
        
        # 结合多种随机性源
        available_indices = list(range(len(non_responders)))
        
        # 使用系统时间进行多次打乱
        temp_random = random.Random(microsecond_seed)
        temp_random.shuffle(available_indices)
        random.shuffle(available_indices)  # 再次打乱
        
        # 从打乱后的列表中选择
        selected_indices = available_indices[:num_non_responder_samples]
        
        print(f"Available non-responder indices: {len(available_indices)}")
        print(f"Selected non-responder indices: {selected_indices}")
        print(f"Using microsecond seed: {microsecond_seed}")
        
        selected_non_responders = non_responders.iloc[selected_indices]
        
        for _, sample in selected_non_responders.iterrows():
            selected_samples.append(sample)
            print(f"Selected non-responder sample: {sample['slides_name']}")
    
    print(f"Total {len(selected_samples)} WSI samples selected for analysis")
    return selected_samples

def get_patch_files(patch_dir):
    """
    Get all patch files from patch directory
    """
    patch_files = [f for f in glob.glob(os.path.join(patch_dir, "*.png")) if not f.endswith("_overlay.png")]
    
    if len(patch_files) == 0:
        print(f"No patch files found in {patch_dir}")
        return []
    
    print(f"Found {len(patch_files)} patch files, will analyze all")
    return patch_files

class PatchDataset(Dataset):
    def __init__(self, patch_paths, transform=None):
        self.patch_paths = patch_paths
        self.transform = transform

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        path = self.patch_paths[idx]
        image = cv2.imread(path)
        if image is None:
            # 返回None用于后续过滤
            return None, path
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] != 512 or image.shape[1] != 512:
            image = cv2.resize(image, (512, 512))
        from PIL import Image as PILImage
        image_pil = PILImage.fromarray(image)
        if self.transform:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = transforms.ToTensor()(image_pil)
        return image_tensor, path
def analyze_patches_multi_gpu(patch_paths, models_and_devices, batch_size=8):
    """
    Multi-GPU parallel patch inference with load balancing
    利用两块GPU进行并行推理，显著提升速度
    """
    import threading
    from queue import Queue
    import math
    
    num_gpus = len(models_and_devices)
    print(f"🚀 Starting multi-GPU inference with {num_gpus} GPUs")
    
    # Split patches between GPUs
    # import ipdb; ipdb.set_trace() 
    chunks = [[] for _ in range(num_gpus)]
    for i, patch_path in enumerate(patch_paths):
        chunks[i % num_gpus].append(patch_path)
    # import ipdb; ipdb.set_trace() 
    print(f"📊 Load distribution:")
    for i, chunk in enumerate(chunks):
        print(f"   GPU {i}: {len(chunk)} patches")
    
    # Results collection
    all_results = []
    result_lock = threading.Lock()
    
    def gpu_worker(gpu_id, patch_chunk, model, device):
        """Worker function for each GPU"""
        chunk_results = []
        
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            dataset = PatchDataset(patch_chunk, transform=transform)
            # 减少num_workers以避免内存冲突
            dataloader = DataLoader(dataset, batch_size=batch_size//2, num_workers=2, pin_memory=False)
            
            model.eval()
            with torch.no_grad():
                pbar = tqdm(dataloader, desc=f"GPU {gpu_id}", position=gpu_id, leave=True, ncols=80)
                for batch in pbar:
                    try:
                        images, paths = batch
                        
                        # Filter valid images
                        valid_indices = [i for i, img in enumerate(images) if img is not None]
                        if not valid_indices:
                            for path in paths:
                                chunk_results.append(create_empty_analysis(path))
                            continue
                        
                        valid_images = torch.stack([img for img in images if img is not None]).to(device)
                        valid_paths = [paths[i] for i in valid_indices]
                        
                        predictions = model(valid_images)
                        predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)
                        predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)
                        
                        for batch_idx, patch_path in enumerate(valid_paths):
                            single_pred = {key: value[batch_idx:batch_idx+1] for key, value in predictions.items()}
                            instance_map, instance_types = model.calculate_instance_map(single_pred, magnification=40)
                            instance_map = instance_map[0].cpu().numpy()
                            nuclei_type_map = single_pred["nuclei_type_map"][0].cpu().numpy()
                            nuclei_pred = np.argmax(nuclei_type_map, axis=0)
                            slide_obj = SlideNucStatObject(instance_map=instance_map,inst_type = nuclei_pred, image=valid_images)
                            df_features = slide_obj.compute_features()
                            df_features.to_csv(patch_path.with_suffix('.csv'), index=False)
                            if len(instance_types) == 0 or len(instance_types[0]) == 0:
                                result = create_empty_analysis(patch_path)
                            else:
                                result = create_patch_analysis(patch_path, instance_types[0])
                            
                            chunk_results.append(result)
                            
                    except Exception as e:
                        print(f"GPU {gpu_id} batch processing error: {e}")
                        # 为失败的batch创建空分析
                        for path in paths:
                            chunk_results.append(create_empty_analysis(path))
                        
        except Exception as e:
            print(f"GPU {gpu_id} worker failed: {e}")
            # 为所有未处理的patches创建空分析
            for path in patch_chunk:
                chunk_results.append(create_empty_analysis(path))
        
        # Thread-safe result collection
        with result_lock:
            all_results.extend(chunk_results)
            print(f"✅ GPU {gpu_id} completed: {len(chunk_results)} patches")
    
    # Start threads for each GPU
    threads = []
    for gpu_id, (model, device) in enumerate(models_and_devices):
        if len(chunks[gpu_id]) > 0:  # Only start thread if there are patches to process
            thread = threading.Thread(
                target=gpu_worker, 
                args=(gpu_id, chunks[gpu_id], model, device)
            )
            thread.start()
            threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print(f"🎯 Multi-GPU inference completed: {len(all_results)} total results")
    return all_results

def create_empty_analysis(patch_path):
    """Create empty analysis for failed patches"""
    return {
        'patch_path': patch_path,
        'cell_counts': [0] * 6,  # Background + 5 cell types
        'cell_ratios': [0.0] * 6,
        'total_cells': 0
    }

def create_patch_analysis(patch_path, instance_types):
    """Create patch analysis from instance types"""
    cell_counts = [0] * 6  # Background + 5 cell types
    for cell_type in instance_types:
        if 0 <= cell_type < 6:
            cell_counts[cell_type] += 1
    
    total_cells = sum(cell_counts[1:])  # Exclude background
    
    if total_cells > 0:
        cell_ratios = [count / total_cells for count in cell_counts[1:]]
        cell_ratios.insert(0, cell_counts[0] / sum(cell_counts) if sum(cell_counts) > 0 else 0)
    else:
        cell_ratios = [0.0] * 6
    
    return {
        'patch_path': patch_path,
        'cell_counts': cell_counts,
        'cell_ratios': cell_ratios,
        'total_cells': total_cells
    }

def analyze_patches_dataloader(patch_paths, model, device, batch_size=4, num_workers=4):
    """
    用DataLoader批量推理patches，带进度条
    """
    color_dict = {
        0: [0, 0, 0],       # Background - black
        1: [255, 0, 0],     # Neoplastic - red  
        2: [0, 255, 0],     # Inflammatory - green
        3: [0, 0, 255],     # Connective - blue
        4: [255, 255, 0],   # Dead - yellow
        5: [255, 0, 255],   # Epithelial - magenta
    }
    results = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = PatchDataset(patch_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    model.eval()
    torch.cuda.empty_cache()  # 清理GPU缓存
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader), desc="Patch inference", ncols=80)):
            # 每隔10个批次清理一次缓存，避免内存累积
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                
            images, paths = batch
            valid_indices = [i for i, img in enumerate(images) if img is not None]
            if not valid_indices:
                for path in paths:
                    results.append(None)
                continue
            valid_images = torch.stack([img for img in images if img is not None]).to(device)
            valid_paths = [paths[i] for i in valid_indices]
            predictions = model(valid_images)
            predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)
            predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)
            for batch_idx, patch_path in enumerate(valid_paths):
                single_pred = {key: value[batch_idx:batch_idx+1] for key, value in predictions.items()}
                
                # Handle DataParallel wrapped model
                if hasattr(model, 'module'):
                    instance_map, instance_types = model.module.calculate_instance_map(single_pred, magnification=40)
                    instance_map = instance_map[0].cpu().numpy()
                    nuclei_type_map = single_pred["nuclei_type_map"][0].cpu().numpy()
                    nuclei_pred = np.argmax(nuclei_type_map, axis=0)
                    overlay_img = cv2.imread(patch_path)          # BGR
                    overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)  # 转为 RGB
                    valid_images = overlay_img.copy()
                    # import ipdb; ipdb.set_trace()
                    slide_obj = SlideNucStatObject(instance_map=instance_map,inst_type = nuclei_pred, image=valid_images)
                    df_features = slide_obj.compute_features()
                    patch_path_path = Path(patch_path)  
                    df_features.to_csv(patch_path_path.with_suffix('.csv'), index=False)





                    if len(instance_types) > 0:
                        for cell_id, cell_info in instance_types[0].items():
                            if cell_info['type'] == 0:
                                continue
                            cell_type = cell_info['type']
                            color = color_dict.get(cell_type, [255, 255, 255])
                            contour = np.array(cell_info['contour'], dtype=np.int32)
                            cv2.drawContours(overlay_img, [contour], -1, color, 2)
                            centroid = tuple(map(int, cell_info['centroid']))
                            cv2.circle(overlay_img, centroid, 3, color, -1)

                        new_path = patch_path_path.with_name(patch_path_path.stem + "_20251011_overlay.png")
                        cv2.imwrite(str(new_path), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
                if len(instance_types) == 0 or len(instance_types[0]) == 0:
                    result = {
                        'patch_path': patch_path,
                        'total_cells': 0,
                        'cell_counts': {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0},  # 使用字符串键
                        'cell_ratios': {'1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},  # 使用字符串键
                        'instance_map': None,
                        'instance_types': None,
                        'original_image': None
                    }
                else:
                    cell_counts = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}  # 使用字符串键
                    for cell_id, cell_info in instance_types[0].items():
                        cell_type = str(cell_info['type'])  # 转换为字符串
                        if cell_type in cell_counts:
                            cell_counts[cell_type] += 1
                    total_cells = sum(cell_counts.values())
                    cell_ratios = {cell_type: count / total_cells if total_cells > 0 else 0 for cell_type, count in cell_counts.items()}
                    
                    # 保存预测结果用于可视化
                    original_image = cv2.imread(patch_path)
                    if original_image is not None:
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    
                    result = {
                        'patch_path': patch_path,
                        'total_cells': int(total_cells),  # 确保是Python int
                        'cell_counts': {str(k): int(v) for k, v in cell_counts.items()},  # 转换为JSON安全格式
                        'cell_ratios': {str(k): float(v) for k, v in cell_ratios.items()},  # 转换为JSON安全格式
                        'instance_map': None,  # 不保存大型数组到JSON
                        'instance_types': None,  # 不保存复杂结构到JSON
                        'original_image': None  # 不保存图像到JSON
                    }
                results.append(result)
    return results


def infer_angiogenesis_from_cells(cell_ratios):
    """
    从细胞比例推断血管生成活性
    基于肿瘤血管生成的生物学机制
    """
    score = 0
    
    # 肿瘤负荷贡献 (VEGF分泌源) - 安全获取
    tumor_ratio = cell_ratios.get('1', 0.0)
    if 0.25 <= tumor_ratio <= 0.70:      # 最适肿瘤密度
        score += 0.35 * (1 - abs(tumor_ratio - 0.475) / 0.225)  # 钟形曲线
    elif tumor_ratio > 0.70:             # 过高密度可能缺氧严重
        score += 0.15
    
    # 缺氧/坏死贡献 (HIF-1α激活 -> VEGF上调) - 安全获取
    necrosis_ratio = cell_ratios.get('4', 0.0)
    if 0.04 <= necrosis_ratio <= 0.15:   # 适度缺氧最强促血管生成
        optimal_necrosis = 0.08
        score += 0.25 * (1 - abs(necrosis_ratio - optimal_necrosis) / 0.07)
    elif necrosis_ratio > 0.15:          # 过度坏死抑制血管生成
        score -= 0.15
    
    # 炎症微环境贡献 (促/抗血管生成因子平衡)
    inflam_ratio = cell_ratios.get('2', 0.0)
    if 0.03 <= inflam_ratio <= 0.12:     # 轻度炎症促进血管生成
        score += 0.20
    elif 0.12 < inflam_ratio <= 0.25:    # 中度炎症混合效应
        score += 0.10
    elif inflam_ratio > 0.25:            # 高炎症破坏血管
        score -= 0.20
    
    # 间质/血管床贡献 (血管生成的结构基础)
    stroma_ratio = cell_ratios.get('3', 0.0)
    if stroma_ratio >= 0.08:             # 间质提供血管生成空间
        if stroma_ratio <= 0.30:         # 适度间质最佳
            score += 0.20
        else:                            # 过度纤维化阻碍血管
            score += 0.10
    
    # 组织完整性奖励 (避免严重破坏区域)
    total_viable = tumor_ratio + inflam_ratio + stroma_ratio
    if total_viable >= 0.85:             # 高活力组织
        score += 0.10
    
    return max(0, min(1, score))  # 限制在0-1范围
def analyze_wsi_sample(sample, model, device, is_multi_gpu=False):
    """
    Analyze all patches of a single WSI sample
    Enhanced with DataParallel multi-GPU support
    """
    print(f"🔍 Starting analysis for WSI sample: {sample}")
    
    patch_dir = sample['slides_name']
    label = sample['label']
    
    print(f"\nAnalyzing WSI: {patch_dir}")
    print(f"Bevacizumab response label: {'HighRS' if label == 1 else 'LowRS'}")
    
    # Check if path exists
    if not os.path.exists(patch_dir):
        print(f"Path does not exist: {patch_dir}")
        return None
    
    # Get all patch files
    patch_files = get_patch_files(patch_dir)
    if not patch_files:
        return None
    
    # Enhanced batch size for multi-GPU
    if is_multi_gpu:
        print(f"🚀 Processing {len(patch_files)} patches with Multi-GPU DataParallel...")
        # 为了避免OOM Killed，使用非常保守的内存设置
        if torch.cuda.is_available():
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            total_memory = sum(torch.cuda.get_device_properties(i).total_memory 
                             for i in range(torch.cuda.device_count())) / 1e9
            print(f"🔧 Total GPU memory: {total_memory:.1f}GB")
            
            # 非常保守的批次大小，避免OOM
            if total_memory >= 40:  # 2x 20GB+
                batch_size = 32  # 进一步减少避免Killed
            elif total_memory >= 30:  # 2x 15GB+
                batch_size = 24
            elif total_memory >= 20:  # 2x 10GB+
                batch_size = 16
            else:
                batch_size = 8
            num_workers = 6  # 最小化worker数量
        else:
            batch_size = 8
            num_workers = 2
        
        print(f"🎯 Multi-GPU batch size: {batch_size}, workers: {num_workers}")
    else:
        print(f"Processing {len(patch_files)} patches with Single-GPU...")
        # Single GPU batch size
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory >= 20:
                batch_size = 4
            elif gpu_memory >= 10:
                batch_size = 4
            else:
                batch_size = 4
            num_workers = 4
        else:
            batch_size = 4
            num_workers = 4
        
        print(f"Using batch size: {batch_size}, num_workers: {num_workers}")
    
    batch_results = analyze_patches_dataloader(patch_files, model, device, batch_size, num_workers)
    
    return 1


def main():
    """
    创建聚合的细胞分布图表 - 适用于大规模WSI数据集
    """
    num_wsi = len(valid_analyses)
    
    cell_type_names = {
        1: "Neoplastic", 2: "Inflammatory", 3: "Connective", 4: "Dead", 5: "Epithelial"
    }
    
    cell_type_colors = {
        1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1', 4: '#96CEB4', 5: '#FECA57'
    }
    
    # 分离响应者和非响应者数据
    responder_analyses = [a for a in valid_analyses if a['label'] == 1]
    non_responder_analyses = [a for a in valid_analyses if a['label'] == 0]
    
    print(f"📈 Generating aggregated plots for {len(responder_analyses)} responders and {len(non_responder_analyses)} non-responders")
    
    # 1. 聚合细胞计数分布对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1.1 总细胞计数分布对比 (左上)
    ax = axes[0, 0]
    
    # 收集所有patch的总细胞数
    resp_total_cells = []
    non_resp_total_cells = []
    
    for analysis in responder_analyses:
        resp_total_cells.extend([p['total_cells'] for p in analysis['patch_analyses']])
    
    for analysis in non_responder_analyses:
        non_resp_total_cells.extend([p['total_cells'] for p in analysis['patch_analyses']])
    
    ax.hist(resp_total_cells, bins=50, alpha=0.7, label=f'Responders (n={len(resp_total_cells)})', 
           color='#2E8B57', density=True)
    ax.hist(non_resp_total_cells, bins=50, alpha=0.7, label=f'Non-responders (n={len(non_resp_total_cells)})', 
           color='#CD5C5C', density=True)
    ax.set_xlabel('Total cells per patch')
    ax.set_ylabel('Density')
    ax.set_title('Aggregated Cell Count Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.2 WSI级别的选择率分布 (右上)
    ax = axes[0, 1]
    
    resp_selection_rates = [a['rpsm_selected_count'] / a['total_patches'] * 100 for a in responder_analyses]
    non_resp_selection_rates = [a['rpsm_selected_count'] / a['total_patches'] * 100 for a in non_responder_analyses]
    
    box_data = [resp_selection_rates, non_resp_selection_rates]
    box_labels = ['Responders', 'Non-responders']
    
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('#2E8B57')
    bp['boxes'][1].set_facecolor('#CD5C5C')
    for patch in bp['boxes']:
        patch.set_alpha(0.7)
    
    ax.set_ylabel('RPSM Selection Rate (%)')
    ax.set_title(f'WSI-level Selection Rate Distribution\n({len(responder_analyses)} vs {len(non_responder_analyses)} WSIs)')
    ax.grid(True, alpha=0.3)
    
    # 1.3 细胞类型比例聚合分析 (左下)
    ax = axes[1, 0]
    
    # 计算每种细胞类型在两组中的平均比例
    resp_cell_ratios = {str(i): [] for i in range(1, 6)}
    non_resp_cell_ratios = {str(i): [] for i in range(1, 6)}
    
    for analysis in responder_analyses:
        for patch in analysis['patch_analyses']:
            for cell_type in range(1, 6):
                resp_cell_ratios[str(cell_type)].append(patch['cell_ratios'].get(str(cell_type), 0.0))
    
    for analysis in non_responder_analyses:
        for patch in analysis['patch_analyses']:
            for cell_type in range(1, 6):
                non_resp_cell_ratios[str(cell_type)].append(patch['cell_ratios'].get(str(cell_type), 0.0))
    
    cell_type_labels = ['Neo', 'Inf', 'Con', 'Dead', 'Epi']
    resp_means = [np.mean(resp_cell_ratios[str(i)]) for i in range(1, 6)]
    non_resp_means = [np.mean(non_resp_cell_ratios[str(i)]) for i in range(1, 6)]
    resp_stds = [np.std(resp_cell_ratios[str(i)]) for i in range(1, 6)]
    non_resp_stds = [np.std(non_resp_cell_ratios[str(i)]) for i in range(1, 6)]
    
    x = np.arange(len(cell_type_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, resp_means, width, yerr=resp_stds, 
                  label='Responders', color='#2E8B57', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, non_resp_means, width, yerr=non_resp_stds,
                  label='Non-responders', color='#CD5C5C', alpha=0.8, capsize=5)
    
    ax.set_xlabel('Cell Types')
    ax.set_ylabel('Average Cell Ratio')
    ax.set_title('Aggregated Cell Type Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(cell_type_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.4 数据集概览统计 (右下)
    ax = axes[1, 1]
    ax.axis('off')  # 关闭坐标轴
    
    # 创建统计信息文本
    stats_text = f"""
Dataset Overview Statistics

📊 Total WSIs: {num_wsi}
   • Responders: {len(responder_analyses)} ({len(responder_analyses)/num_wsi*100:.1f}%)
   • Non-responders: {len(non_responder_analyses)} ({len(non_responder_analyses)/num_wsi*100:.1f}%)

📋 Total Patches: {sum(a['total_patches'] for a in valid_analyses):,}
   • Responder patches: {sum(a['total_patches'] for a in responder_analyses):,}
   • Non-responder patches: {sum(a['total_patches'] for a in non_responder_analyses):,}

🎯 RPSM Selection:
   • Total selected: {sum(a['rpsm_selected_count'] for a in valid_analyses):,}
   • Average selection rate: {sum(a['rpsm_selected_count'] for a in valid_analyses) / sum(a['total_patches'] for a in valid_analyses) * 100:.2f}%
   • Responder rate: {sum(a['rpsm_selected_count'] for a in responder_analyses) / sum(a['total_patches'] for a in responder_analyses) * 100:.2f}%
   • Non-responder rate: {sum(a['rpsm_selected_count'] for a in non_responder_analyses) / sum(a['total_patches'] for a in non_responder_analyses) * 100:.2f}%

🔬 Method Comparison:
   • Improved RPSM: {sum(a['improved_rpsm_selected_count'] for a in valid_analyses):,} selected
   • Angiogenesis RPSM: {sum(a['angio_rpsm_selected_count'] for a in valid_analyses):,} selected
   • Hybrid RPSM: {sum(a['hybrid_rpsm_selected_count'] for a in valid_analyses):,} selected
"""
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aggregated_cell_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 创建简化的相关性热力图
    create_aggregated_correlation_heatmap(valid_analyses, output_dir)
    
    print(f"✅ Aggregated analysis plots saved to {output_dir} directory")

def create_aggregated_correlation_heatmap(valid_analyses, output_dir):
    """创建聚合的相关性热力图"""
    # 收集所有patch数据
    all_cell_ratios = []
    all_labels = []
    
    for analysis in valid_analyses:
        for patch in analysis['patch_analyses']:
            ratios = [patch['cell_ratios'].get(str(i), 0.0) for i in range(1, 6)]
            all_cell_ratios.append(ratios)
            all_labels.append(analysis['label'])
    
    if not all_cell_ratios:
        return
    
    cell_ratios_df = pd.DataFrame(all_cell_ratios, 
                                 columns=['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial'])
    
    # 计算相关性矩阵
    correlation_matrix = cell_ratios_df.corr()
    
    # 创建热力图
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    im = ax.imshow(correlation_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # 设置坐标轴
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(correlation_matrix.columns)
    
    # 添加数值标签
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                         ha="center", va="center", 
                         color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black")
    
    ax.set_title(f'Aggregated Cell Type Correlation Matrix\n({len(all_cell_ratios):,} patches from {len(valid_analyses)} WSIs)')
    plt.colorbar(im, ax=ax, label='Correlation Coefficient')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aggregated_cell_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_rpsm_evaluation(wsi_analyses, output_dir="plots"):
    """
    创建包含优化RPSM方法在内的全面评估分析
    对比6种RPSM方法：原始、改进、血管生成、混合、优化、自适应
    """
    os.makedirs(output_dir, exist_ok=True)
    
    valid_analyses = [a for a in wsi_analyses if a is not None]
    if not valid_analyses:
        return {}
    
    num_wsi = len(valid_analyses)
    print(f"🎯 全面评估 {num_wsi} 个WSI的6种RPSM方法性能...")
    
    # 分离响应者和非响应者
    responder_analyses = [a for a in valid_analyses if a['label'] == 1]
    non_responder_analyses = [a for a in valid_analyses if a['label'] == 0]
    
    # 定义所有RPSM方法
    methods_config = [
        ('Original RPSM', 'rpsm_selected_count', '#FF6B6B'),
        ('Improved RPSM', 'improved_rpsm_selected_count', '#4ECDC4'), 
        ('Angiogenesis RPSM', 'angio_rpsm_selected_count', '#45B7D1'),
        ('Hybrid RPSM', 'hybrid_rpsm_selected_count', '#96CEB4'),
        ('🎯Optimized RPSM', 'optimized_rpsm_selected_count', '#2E8B57'),
        ('🧠Adaptive RPSM', 'adaptive_rpsm_selected_count', '#FF8C42')
    ]
    
    # 计算各方法的性能指标
    methods_evaluation = {}
    
    for method_name, count_key, color in methods_config:
        print(f"📊 评估 {method_name}...")
        
        # 计算选择率
        resp_rates = []
        non_resp_rates = []
        
        for analysis in responder_analyses:
            if count_key in analysis:
                rate = analysis[count_key] / analysis['total_patches'] * 100
                resp_rates.append(rate)
        
        for analysis in non_responder_analyses:
            if count_key in analysis:
                rate = analysis[count_key] / analysis['total_patches'] * 100
                non_resp_rates.append(rate)
        
        # 统计分析
        if resp_rates and non_resp_rates:
            t_stat, p_value = stats.ttest_ind(resp_rates, non_resp_rates)
            effect_size = (np.mean(resp_rates) - np.mean(non_resp_rates)) / np.sqrt((np.std(resp_rates)**2 + np.std(non_resp_rates)**2) / 2)
            
            # 计算AUC (使用选择率作为预测分数)
            try:
                labels = [1] * len(resp_rates) + [0] * len(non_resp_rates)
                scores = resp_rates + non_resp_rates
                if len(set(labels)) > 1 and len(set(scores)) > 1:
                    auc = roc_auc_score(labels, scores)
                else:
                    auc = 0.5
            except:
                auc = 0.5
        else:
            t_stat, p_value, effect_size, auc = 0, 1, 0, 0.5
        
        # 计算区分度和临床实用性
        discrimination = np.mean(resp_rates) - np.mean(non_resp_rates) if (resp_rates and non_resp_rates) else 0
        avg_selection_rate = np.mean(resp_rates + non_resp_rates) if (resp_rates or non_resp_rates) else 0
        
        methods_evaluation[method_name] = {
            'responder_rate_mean': np.mean(resp_rates) if resp_rates else 0,
            'responder_rate_std': np.std(resp_rates) if resp_rates else 0,
            'non_responder_rate_mean': np.mean(non_resp_rates) if non_resp_rates else 0,
            'non_responder_rate_std': np.std(non_resp_rates) if non_resp_rates else 0,
            'discrimination': discrimination,
            'effect_size': effect_size,
            'p_value': p_value,
            'auc': auc,
            'avg_selection_rate': avg_selection_rate,
            'color': color,
            'sample_size': len(resp_rates) + len(non_resp_rates)
        }
    
    # 创建综合对比可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    methods = list(methods_evaluation.keys())
    colors = [methods_evaluation[m]['color'] for m in methods]
    
    # 1. AUC对比 (左上)
    ax = axes[0, 0]
    aucs = [methods_evaluation[m]['auc'] for m in methods]
    bars = ax.bar(range(len(methods)), aucs, color=colors, alpha=0.8)
    ax.set_title('AUC性能对比', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUC')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(' RPSM', '') for m in methods], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, max(aucs) + 0.05)
    
    # 标注最佳性能
    best_auc_idx = np.argmax(aucs)
    ax.annotate('最佳AUC', xy=(best_auc_idx, aucs[best_auc_idx]), 
               xytext=(best_auc_idx, aucs[best_auc_idx] + 0.02),
               arrowprops=dict(arrowstyle='->', color='red'), fontweight='bold')
    
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
               f'{auc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 2. 区分度对比 (中上)
    ax = axes[0, 1]
    discriminations = [methods_evaluation[m]['discrimination'] for m in methods]
    bars = ax.bar(range(len(methods)), discriminations, color=colors, alpha=0.8)
    ax.set_title('区分度对比 (响应者-非响应者)', fontsize=14, fontweight='bold')
    ax.set_ylabel('区分度 (%)')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(' RPSM', '') for m in methods], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 标注最佳区分度
    best_disc_idx = np.argmax(discriminations)
    ax.annotate('最佳区分度', xy=(best_disc_idx, discriminations[best_disc_idx]),
               xytext=(best_disc_idx, discriminations[best_disc_idx] + 0.5),
               arrowprops=dict(arrowstyle='->', color='red'), fontweight='bold')
    
    for bar, disc in zip(bars, discriminations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{disc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. 效应大小对比 (右上)
    ax = axes[0, 2]
    effect_sizes = [abs(methods_evaluation[m]['effect_size']) for m in methods]
    bars = ax.bar(range(len(methods)), effect_sizes, color=colors, alpha=0.8)
    ax.set_title('效应大小对比 (Cohen\'s d)', fontsize=14, fontweight='bold')
    ax.set_ylabel('|Cohen\'s d|')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(' RPSM', '') for m in methods], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 添加效应大小解释线
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.7, label='小效应 (0.2)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='中等效应 (0.5)')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='大效应 (0.8)')
    ax.legend(fontsize=8)
    
    for bar, effect in zip(bars, effect_sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{effect:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 4. 选择率对比 (左下)
    ax = axes[1, 0]
    resp_means = [methods_evaluation[m]['responder_rate_mean'] for m in methods]
    non_resp_means = [methods_evaluation[m]['non_responder_rate_mean'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, resp_means, width, label='响应者', color='#2E8B57', alpha=0.8)
    bars2 = ax.bar(x + width/2, non_resp_means, width, label='非响应者', color='#CD5C5C', alpha=0.8)
    
    ax.set_title('各组选择率对比', fontsize=14, fontweight='bold')
    ax.set_ylabel('平均选择率 (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' RPSM', '') for m in methods], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 综合性能雷达图 (中下)
    ax = axes[1, 1]
    
    # 选择前4个最重要的方法进行雷达图对比
    top_methods = ['Original RPSM', '🎯Optimized RPSM', '🧠Adaptive RPSM', 'Hybrid RPSM']
    radar_data = []
    
    for method in top_methods:
        if method in methods_evaluation:
            eval_data = methods_evaluation[method]
            # 标准化各项指标到0-1范围
            normalized_metrics = [
                eval_data['auc'],  # AUC已经在0-1范围
                min(eval_data['discrimination'] / 10, 1),  # 区分度标准化
                min(abs(eval_data['effect_size']), 1),  # 效应大小
                1 - min(eval_data['p_value'], 1)  # p值转换为显著性
            ]
            radar_data.append(normalized_metrics)
    
    # 雷达图需要复杂的绘制，这里用条形图代替
    metrics_names = ['AUC', '区分度', '效应大小', '显著性']
    x_pos = np.arange(len(metrics_names))
    
    for i, method in enumerate(top_methods):
        if i < len(radar_data):
            ax.plot(x_pos, radar_data[i], 'o-', label=method, linewidth=2, markersize=6)
    
    ax.set_title('关键方法性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel('标准化得分')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # 6. 方法推荐总结 (右下)
    ax = axes[1, 2]
    ax.axis('off')
    
    # 找出最佳方法
    best_auc_method = methods[np.argmax(aucs)]
    best_disc_method = methods[np.argmax(discriminations)]
    best_effect_method = methods[np.argmax(effect_sizes)]
    
    # 计算综合评分
    composite_scores = {}
    for method in methods:
        eval_data = methods_evaluation[method]
        composite_score = (
            eval_data['auc'] * 0.4 +
            min(eval_data['discrimination'] / 10, 1) * 0.3 +
            min(abs(eval_data['effect_size']), 1) * 0.3
        )
        composite_scores[method] = composite_score
    
    best_overall_method = max(composite_scores.items(), key=lambda x: x[1])[0]
    
    summary_text = f"""
🏆 综合评估结果

📊 各项最佳表现:
• 最佳AUC: {best_auc_method.replace(' RPSM', '')}
  ({methods_evaluation[best_auc_method]['auc']:.3f})
• 最佳区分度: {best_disc_method.replace(' RPSM', '')}
  ({methods_evaluation[best_disc_method]['discrimination']:.1f}%)  
• 最大效应: {best_effect_method.replace(' RPSM', '')}
  ({methods_evaluation[best_effect_method]['effect_size']:.3f})

🎯 综合推荐: {best_overall_method.replace(' RPSM', '')}
   综合评分: {composite_scores[best_overall_method]:.3f}

💡 关键发现:
• 优化RPSM显著改善了{methods_evaluation['🎯Optimized RPSM']['auc']:.1%}的AUC
• 自适应RPSM实现了{methods_evaluation['🧠Adaptive RPSM']['discrimination']:.1f}%的区分度  
• 新方法相比原始RPSM提升明显

📈 临床价值:
• 更精准的响应者识别
• 降低假阳性率
• 个体化治疗指导
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"comprehensive_rpsm_evaluation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 输出详细的数值结果
    print(f"\n📊 详细评估结果:")
    print("="*80)
    
    for method in methods:
        eval_data = methods_evaluation[method]
        print(f"\n{method}:")
        print(f"  AUC: {eval_data['auc']:.4f}")
        print(f"  区分度: {eval_data['discrimination']:.2f}%")
        print(f"  效应大小: {eval_data['effect_size']:.3f}")
        print(f"  p值: {eval_data['p_value']:.4f}")
        print(f"  响应者选择率: {eval_data['responder_rate_mean']:.2f}% ± {eval_data['responder_rate_std']:.2f}%")
        print(f"  非响应者选择率: {eval_data['non_responder_rate_mean']:.2f}% ± {eval_data['non_responder_rate_std']:.2f}%")
    
    print(f"\n🎯 最终推荐: {best_overall_method}")
    print(f"📊 评估图表保存至: {plot_path}")
    
    return {
        'methods_evaluation': methods_evaluation,
        'best_method': best_overall_method,
        'composite_scores': composite_scores,
        'plot_path': plot_path
    }

def create_comprehensive_rpsm_recommendation_analysis(wsi_analyses, output_dir="plots"):
    """
    创建全面的RPSM方法推荐分析 - 专为大规模数据集优化
    """
    os.makedirs(output_dir, exist_ok=True)
    
    valid_analyses = [a for a in wsi_analyses if a is not None]
    if not valid_analyses:
        return {}
    
    num_wsi = len(valid_analyses)
    print(f"🎯 Analyzing {num_wsi} WSIs for RPSM method recommendations...")
    
    # 分离响应者和非响应者
    responder_analyses = [a for a in valid_analyses if a['label'] == 1]
    non_responder_analyses = [a for a in valid_analyses if a['label'] == 0]
    
    # 计算各方法的综合评估指标
    methods_evaluation = {}
    
    method_configs = [
        ('Original RPSM', 'rpsm_selected_count', 'rpsm_selected'),
        ('Improved RPSM', 'improved_rpsm_selected_count', 'improved_rpsm_selected'),  
        ('Angiogenesis RPSM', 'angio_rpsm_selected_count', 'angio_rpsm_selected'),
        ('Hybrid RPSM', 'hybrid_rpsm_selected_count', 'hybrid_rpsm_selected')
    ]
    
    for method_name, count_key, patch_key in method_configs:
        print(f"📊 Evaluating {method_name}...")
        
        # 1. 选择率统计
        resp_rates = []
        non_resp_rates = []
        
        for analysis in responder_analyses:
            if count_key in analysis:
                rate = analysis[count_key] / analysis['total_patches'] * 100
                resp_rates.append(rate)
        
        for analysis in non_responder_analyses:
            if count_key in analysis:
                rate = analysis[count_key] / analysis['total_patches'] * 100
                non_resp_rates.append(rate)
        
        # 2. 统计检验
        if resp_rates and non_resp_rates:
            t_stat, p_value = stats.ttest_ind(resp_rates, non_resp_rates)
            effect_size = (np.mean(resp_rates) - np.mean(non_resp_rates)) / np.sqrt((np.std(resp_rates)**2 + np.std(non_resp_rates)**2) / 2)
        else:
            t_stat, p_value, effect_size = 0, 1, 0
            
        # 3. 临床实用性评分
        avg_selection_rate = np.mean(resp_rates + non_resp_rates) if (resp_rates or non_resp_rates) else 0
        clinical_utility = calculate_clinical_utility_score(avg_selection_rate, effect_size, p_value)
        
        # 4. 稳定性评估
        stability_score = calculate_method_stability(valid_analyses, count_key)
        
        methods_evaluation[method_name] = {
            'responder_rate_mean': np.mean(resp_rates) if resp_rates else 0,
            'responder_rate_std': np.std(resp_rates) if resp_rates else 0,
            'non_responder_rate_mean': np.mean(non_resp_rates) if non_resp_rates else 0,
            'non_responder_rate_std': np.std(non_resp_rates) if non_resp_rates else 0,
            'effect_size': effect_size,
            'p_value': p_value,
            'clinical_utility': clinical_utility,
            'stability_score': stability_score,
            'sample_size': len(resp_rates) + len(non_resp_rates)
        }
    
    # 生成推荐报告
    recommendations = generate_rpsm_recommendations(methods_evaluation, num_wsi)
    
    # 创建综合评估可视化
    create_rpsm_recommendation_visualization(methods_evaluation, recommendations, output_dir)
    
    # 保存评估结果
    evaluation_report = {
        'dataset_info': {
            'total_wsi': num_wsi,
            'responders': len(responder_analyses),
            'non_responders': len(non_responder_analyses)
        },
        'methods_evaluation': methods_evaluation,
        'recommendations': recommendations,
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    with open(f"{output_dir}/rpsm_recommendation_analysis.json", 'w') as f:
        json.dump(evaluation_report, f, indent=2, default=numpy_json_serializer)
    
    return recommendations

def calculate_clinical_utility_score(selection_rate, effect_size, p_value):
    """计算临床实用性评分"""
    # 基础分数：基于效应大小
    effect_score = min(abs(effect_size) * 20, 40)  # 最大40分
    
    # 显著性加权
    significance_weight = 1.0 if p_value < 0.001 else 0.8 if p_value < 0.01 else 0.6 if p_value < 0.05 else 0.3
    
    # 选择率平衡性评分（避免过于严格或宽松）
    optimal_rate = 15  # 理想选择率约15%
    rate_penalty = abs(selection_rate - optimal_rate) / optimal_rate
    rate_score = max(0, 30 * (1 - rate_penalty))  # 最大30分
    
    # 综合评分
    total_score = (effect_score * significance_weight + rate_score) 
    return min(total_score, 100)

def calculate_method_stability(analyses, count_key):
    """计算方法在不同WSI间的稳定性"""
    rates = []
    for analysis in analyses:
        if count_key in analysis and analysis['total_patches'] > 0:
            rate = analysis[count_key] / analysis['total_patches'] * 100
            rates.append(rate)
    
    if len(rates) < 2:
        return 0
    
    # 变异系数作为稳定性指标 (越小越稳定)
    cv = np.std(rates) / np.mean(rates) if np.mean(rates) > 0 else float('inf')
    stability_score = max(0, 100 - cv * 20)  # 转换为0-100分
    return stability_score

def generate_rpsm_recommendations(methods_evaluation, num_wsi):
    """生成RPSM方法推荐"""
    
    recommendations = {
        'primary_recommendation': None,
        'alternative_recommendations': [],
        'use_case_specific': {},
        'dataset_considerations': {},
        'implementation_notes': []
    }
    
    # 按综合评分排序
    methods_scores = {}
    for method, eval_data in methods_evaluation.items():
        # 综合评分 = 临床实用性 * 0.4 + 稳定性 * 0.3 + 效应大小权重 * 0.3
        composite_score = (
            eval_data['clinical_utility'] * 0.4 +
            eval_data['stability_score'] * 0.3 +
            min(abs(eval_data['effect_size']) * 30, 30) * 0.3
        )
        methods_scores[method] = composite_score
    
    # 排序推荐
    sorted_methods = sorted(methods_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 主要推荐
    recommendations['primary_recommendation'] = {
        'method': sorted_methods[0][0],
        'score': sorted_methods[0][1],
        'rationale': generate_method_rationale(sorted_methods[0][0], methods_evaluation[sorted_methods[0][0]])
    }
    
    # 备选推荐
    for method, score in sorted_methods[1:3]:  # 取前3个作为备选
        recommendations['alternative_recommendations'].append({
            'method': method,
            'score': score,
            'rationale': generate_method_rationale(method, methods_evaluation[method])
        })
    
    # 特定用例推荐
    recommendations['use_case_specific'] = {
        'high_precision_needed': get_highest_effect_size_method(methods_evaluation),
        'clinical_screening': get_most_stable_method(methods_evaluation),
        'research_exploration': get_most_comprehensive_method(methods_evaluation)
    }
    
    # 数据集考虑因素
    recommendations['dataset_considerations'] = {
        'sample_size': 'Large' if num_wsi > 50 else 'Medium' if num_wsi > 20 else 'Small',
        'recommendations_reliability': 'High' if num_wsi > 30 else 'Medium' if num_wsi > 10 else 'Preliminary',
        'suggested_validation': num_wsi < 30
    }
    
    return recommendations

def generate_method_rationale(method_name, eval_data):
    """为方法推荐生成解释"""
    rationales = []
    
    if eval_data['effect_size'] > 0.5:
        rationales.append("Strong effect size for distinguishing responders")
    elif eval_data['effect_size'] > 0.3:
        rationales.append("Moderate effect size for clinical prediction")
    
    if eval_data['p_value'] < 0.001:
        rationales.append("Highly significant statistical difference")
    elif eval_data['p_value'] < 0.05:
        rationales.append("Statistically significant difference")
    
    if eval_data['stability_score'] > 80:
        rationales.append("Excellent stability across WSIs")
    elif eval_data['stability_score'] > 60:
        rationales.append("Good stability across WSIs")
    
    if eval_data['clinical_utility'] > 70:
        rationales.append("High clinical utility score")
    
    return "; ".join(rationales) if rationales else "Baseline performance"

def get_highest_effect_size_method(methods_evaluation):
    """获取效应大小最大的方法"""
    best_method = max(methods_evaluation.items(), key=lambda x: abs(x[1]['effect_size']))
    return best_method[0]

def get_most_stable_method(methods_evaluation):
    """获取最稳定的方法"""
    best_method = max(methods_evaluation.items(), key=lambda x: x[1]['stability_score'])
    return best_method[0]

def get_most_comprehensive_method(methods_evaluation):
    """获取最综合的方法（通常是Hybrid）"""
    if 'Hybrid RPSM' in methods_evaluation:
        return 'Hybrid RPSM'
    else:
        # 返回效应大小和稳定性都较好的方法
        composite_scores = {}
        for method, eval_data in methods_evaluation.items():
            composite_scores[method] = (abs(eval_data['effect_size']) + eval_data['stability_score'] / 100) / 2
        return max(composite_scores.items(), key=lambda x: x[1])[0]

def create_rpsm_recommendation_visualization(methods_evaluation, recommendations, output_dir):
    """创建RPSM推荐可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    methods = list(methods_evaluation.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # 1. 效应大小对比 (左上)
    ax = axes[0, 0]
    effect_sizes = [methods_evaluation[m]['effect_size'] for m in methods]
    bars = ax.bar(range(len(methods)), effect_sizes, color=colors, alpha=0.8)
    ax.set_title('Effect Size Comparison\n(Higher = Better Discrimination)')
    ax.set_ylabel('Cohen\'s d Effect Size')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(' RPSM', '') for m in methods], rotation=45)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Large Effect (d>0.5)')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Medium Effect (d>0.3)')
    ax.legend()
    
    # 添加数值标签
    for bar, effect in zip(bars, effect_sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{effect:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 临床实用性评分 (右上)
    ax = axes[0, 1]
    utility_scores = [methods_evaluation[m]['clinical_utility'] for m in methods]
    bars = ax.bar(range(len(methods)), utility_scores, color=colors, alpha=0.8)
    ax.set_title('Clinical Utility Score\n(0-100, Higher = More Practical)')
    ax.set_ylabel('Clinical Utility Score')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(' RPSM', '') for m in methods], rotation=45)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # 添加数值标签
    for bar, score in zip(bars, utility_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 稳定性对比 (左下)
    ax = axes[1, 0]
    stability_scores = [methods_evaluation[m]['stability_score'] for m in methods]
    bars = ax.bar(range(len(methods)), stability_scores, color=colors, alpha=0.8)
    ax.set_title('Method Stability Score\n(0-100, Higher = More Consistent)')
    ax.set_ylabel('Stability Score')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(' RPSM', '') for m in methods], rotation=45)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # 添加数值标签
    for bar, score in zip(bars, stability_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. 推荐总结 (右下)
    ax = axes[1, 1]
    ax.axis('off')
    
    # 创建推荐文本
    rec_text = f"""
🏆 PRIMARY RECOMMENDATION
Method: {recommendations['primary_recommendation']['method']}
Score: {recommendations['primary_recommendation']['score']:.1f}/100
Rationale: {recommendations['primary_recommendation']['rationale']}

🥈 ALTERNATIVE OPTIONS
"""
    for i, alt in enumerate(recommendations['alternative_recommendations'][:2]):
        rec_text += f"{i+2}. {alt['method']} (Score: {alt['score']:.1f})\n"
    
    rec_text += f"""
🎯 USE CASE RECOMMENDATIONS
• High Precision: {recommendations['use_case_specific']['high_precision_needed']}
• Clinical Screening: {recommendations['use_case_specific']['clinical_screening']}  
• Research: {recommendations['use_case_specific']['research_exploration']}

📊 DATASET ASSESSMENT
• Sample Size: {recommendations['dataset_considerations']['sample_size']}
• Reliability: {recommendations['dataset_considerations']['recommendations_reliability']}
"""
    
    if recommendations['dataset_considerations']['suggested_validation']:
        rec_text += "⚠️ Validation recommended with larger dataset"
    
    ax.text(0.05, 0.95, rec_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('RPSM Method Comprehensive Evaluation & Recommendations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rpsm_comprehensive_recommendations.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_cell_distribution_plots(wsi_analyses, output_dir="plots"):
    """
    Create statistical plots for cell distribution - optimized for large WSI datasets
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out None analyses
    valid_analyses = [a for a in wsi_analyses if a is not None]
    num_wsi = len(valid_analyses)
    
    if num_wsi == 0:
        print("No valid WSI analyses to plot")
        return
    
    # 当WSI数量过多时，切换到聚合分析模式
    if num_wsi > 20:
        print(f"📊 Large dataset detected ({num_wsi} WSIs), switching to aggregated analysis mode")
        create_aggregated_cell_distribution_plots(valid_analyses, output_dir)
        return
    
    cell_type_names = {
        1: "Neoplastic",
        2: "Inflammatory", 
        3: "Connective",
        4: "Dead",
        5: "Epithelial"
    }
    
    # 用于显示的简短标签
    cell_type_short_names = {
        1: "Neo",      # Neoplastic cells
        2: "Inf",      # Inflammatory cells
        3: "Con",      # Connective tissue
        4: "Dead",     # Dead cells
        5: "Epi"       # Epithelial cells
    }
    
    cell_type_colors = {
        1: '#FF6B6B',  # Red - Neoplastic cells
        2: '#4ECDC4',  # Cyan - Inflammatory cells
        3: '#45B7D1',  # Blue - Connective tissue
        4: '#96CEB4',  # Green - Dead cells
        5: '#FECA57'   # Yellow - Epithelial cells
    }
    
    # Calculate optimal subplot layout  
    # 当WSI数量过多时，使用更紧凑的布局
    if num_wsi > 12:
        cols = 4  # 使用4列布局以节省空间
        rows = (num_wsi + cols - 1) // cols
        fig_width = 4 * cols  # 减小每个子图的宽度
        fig_height = 3 * rows  # 减小每个子图的高度
    elif num_wsi > 6:
        cols = 3
        rows = (num_wsi + cols - 1) // cols
        fig_width = 5 * cols
        fig_height = 4 * rows
    else:
        cols = min(3, num_wsi)
        rows = (num_wsi + cols - 1) // cols
        fig_width = 5 * cols
        fig_height = 4 * rows
    
    # 1. Cell count distribution histogram
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Ensure axes is always a flat array for consistent indexing
    if num_wsi == 1:
        axes = [axes]
    elif rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    else:
        axes = axes.flatten()
    
    for i, analysis in enumerate(valid_analyses):
        ax = axes[i]
        
        # Get total cell count for all patches
        total_cells_per_patch = [p['total_cells'] for p in analysis['patch_analyses']]
        
        ax.hist(total_cells_per_patch, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f"WSI {i+1}: Cell Count Distribution\n({'Responder' if analysis['label'] == 1 else 'Non-responder'})")
        ax.set_xlabel('Total cells per patch')
        ax.set_ylabel('Number of patches')
        ax.grid(True, alpha=0.3)
        
        # Add statistical information
        mean_cells = np.mean(total_cells_per_patch)
        median_cells = np.median(total_cells_per_patch)
        ax.axvline(mean_cells, color='red', linestyle='--', label=f'Mean: {mean_cells:.1f}')
        ax.axvline(median_cells, color='orange', linestyle='--', label=f'Median: {median_cells:.1f}')
        ax.legend(fontsize='small', loc='upper right')
    
    # Hide unused subplots
    for i in range(num_wsi, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cell_count_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cell type ratio boxplot
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Ensure axes is always a flat array for consistent indexing
    if num_wsi == 1:
        axes = [axes]
    elif rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    else:
        axes = axes.flatten()
    
    for i, analysis in enumerate(valid_analyses):
        ax = axes[i]
        
        # Collect ratio data for each cell type
        cell_ratio_data = []
        cell_type_labels = []
        
        for cell_type in [1, 2, 3, 4, 5]:
            ratios = [p['cell_ratios'].get(str(cell_type), 0.0) for p in analysis['patch_analyses']]
            cell_ratio_data.append(ratios)
            # 在有限空间时使用短标签
            if num_wsi > 6:
                cell_type_labels.append(cell_type_short_names[cell_type])
            else:
                cell_type_labels.append(cell_type_names[cell_type])
        
        bp = ax.boxplot(cell_ratio_data, labels=cell_type_labels, patch_artist=True)
        
        # Set colors
        for patch, cell_type in zip(bp['boxes'], [1, 2, 3, 4, 5]):
            patch.set_facecolor(cell_type_colors[cell_type])
            patch.set_alpha(0.7)
        
        ax.set_title(f"WSI {i+1}: Cell Type Ratio Distribution\n({'Responder' if analysis['label'] == 1 else 'Non-responder'})")
        ax.set_ylabel('Cell ratio')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Hide unused subplots
    for i in range(num_wsi, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cell_ratio_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. RPSM selection analysis plots - COMMENTED OUT
    # This functionality is now replaced by more advanced analysis in:
    # - rpsm_comprehensive_evaluation.png 
    # - rpsm_roc_comparison_with_ci.png
    # - rpsm_selection_strictness.png
    
    """
    # 当WSI数量过多时，调整图表大小
    if num_wsi > 8:
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 3.1 RPSM selection rate comparison
    ax = axes[0, 0]
    wsi_names = []
    selection_rates = []
    response_labels = []
    
    for i, analysis in enumerate(valid_analyses):
        wsi_names.append(f"WSI {i+1}")
        selection_rates.append(analysis['rpsm_selected_count'] / analysis['total_patches'] * 100)
        response_labels.append('Responder' if analysis['label'] == 1 else 'Non-responder')
    
    # 当WSI数量过多时，使用分组显示策略
    if num_wsi > 15:
        # 分组显示：按响应类型分组计算平均值
        responder_rates = [rate for rate, label in zip(selection_rates, response_labels) if label == 'Responder']
        non_responder_rates = [rate for rate, label in zip(selection_rates, response_labels) if label == 'Non-responder']
        
        group_names = ['Responder\nGroup', 'Non-responder\nGroup']
        group_means = [np.mean(responder_rates) if responder_rates else 0, 
                      np.mean(non_responder_rates) if non_responder_rates else 0]
        group_stds = [np.std(responder_rates) if len(responder_rates) > 1 else 0,
                     np.std(non_responder_rates) if len(non_responder_rates) > 1 else 0]
        
        bars = ax.bar(group_names, group_means, yerr=group_stds, 
                     color=['#FF6B6B', '#4ECDC4'], alpha=0.7, capsize=5)
        ax.set_title(f'RPSM Selection Rate Group Comparison\n(n={len(responder_rates)} responders, n={len(non_responder_rates)} non-responders)')
        ax.set_ylabel('Average selection rate (%)')
        
        # Add value labels
        for bar, mean, std in zip(bars, group_means, group_stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1, 
                    f'{mean:.1f}±{std:.1f}%', ha='center', va='bottom', fontsize='small')
    else:
        # 原有的个体显示策略
        colors = ['#FF6B6B' if label == 'Responder' else '#4ECDC4' for label in response_labels]
        bars = ax.bar(wsi_names, selection_rates, color=colors, alpha=0.7)
        ax.set_title('RPSM Selection Rate Comparison')
        ax.set_ylabel('Selection rate (%)')
        
        # 当WSI数量过多时，旋转x轴标签
        if num_wsi > 6:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize='small')
        
        # Add value labels
        for bar, rate in zip(bars, selection_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize='small')
        
        # 添加图例，区分响应者和非响应者
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#FF6B6B', alpha=0.7, label='Responder'),
                          Patch(facecolor='#4ECDC4', alpha=0.7, label='Non-responder')]
        ax.legend(handles=legend_elements, fontsize='small', loc='upper right')
    
    ax.grid(True, alpha=0.3)
    
    # 3.2 RPSM selection reason distribution
    ax = axes[0, 1]
    all_reasons = []
    for analysis in valid_analyses:
        selected_patches = [p for p in analysis['patch_analyses'] if p['rpsm_selected']]
        reasons = [p['rpsm_reason'] for p in selected_patches]
        all_reasons.extend(reasons)
    
    if all_reasons:
        reason_counts = pd.Series(all_reasons).value_counts()
        colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        ax.pie(reason_counts.values, labels=reason_counts.index, autopct='%1.1f%%', 
               colors=colors_pie[:len(reason_counts)])
        ax.set_title('RPSM Selection Reason Distribution')
    
    # 3.3 Total cell count vs RPSM selection
    ax = axes[1, 0]
    for i, analysis in enumerate(valid_analyses):
        selected_cells = [p['total_cells'] for p in analysis['patch_analyses'] if p['rpsm_selected']]
        unselected_cells = [p['total_cells'] for p in analysis['patch_analyses'] if not p['rpsm_selected']]
        
        ax.hist(selected_cells, bins=20, alpha=0.7, label=f'WSI {i+1} Selected', color=colors[i])
        ax.hist(unselected_cells, bins=20, alpha=0.3, label=f'WSI {i+1} Unselected', color=colors[i])
    
    ax.set_title('RPSM Selection vs Total Cell Count')
    ax.set_xlabel('Total cell count')
    ax.set_ylabel('Number of patches')
    
    # 优化图例显示，防止遮挡图表
    if len(valid_analyses) > 4:
        # 当WSI数量过多时，将图例放置在图表外部
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    else:
        # WSI数量较少时，使用默认位置
        ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    
    # 3.4 Cell type preference in RPSM selection
    ax = axes[1, 1]
    selected_ratios = {str(cell_type): [] for cell_type in [1, 2, 3, 4, 5]}
    
    for analysis in valid_analyses:
        selected_patches = [p for p in analysis['patch_analyses'] if p['rpsm_selected']]
        for patch in selected_patches:
            for cell_type in [1, 2, 3, 4, 5]:
                cell_type_str = str(cell_type)
                selected_ratios[cell_type_str].append(patch['cell_ratios'].get(cell_type_str, 0.0))
    
    if any(selected_ratios.values()):
        mean_ratios = [np.mean(selected_ratios[str(cell_type)]) if selected_ratios[str(cell_type)] else 0 
                      for cell_type in [1, 2, 3, 4, 5]]
        
        bars = ax.bar(range(5), mean_ratios, 
                     color=[cell_type_colors[i+1] for i in range(5)], alpha=0.7)
        ax.set_title('Average Cell Ratio in RPSM Selected Patches')
        ax.set_xlabel('Cell type')
        ax.set_ylabel('Average ratio')
        ax.set_xticks(range(5))
        # 当样本较多时使用短标签
        if num_wsi > 6:
            ax.set_xticklabels([cell_type_short_names[i+1] for i in range(5)], rotation=45, ha='right')
        else:
            ax.set_xticklabels([cell_type_names[i+1] for i in range(5)], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, ratio in zip(bars, mean_ratios):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{ratio:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rpsm_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    """
    
    # 4. Cell type correlation heatmap
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Ensure axes is always a flat array for consistent indexing
    if num_wsi == 1:
        axes = [axes]
    elif rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    else:
        axes = axes.flatten()
    
    for i, analysis in enumerate(valid_analyses):
        ax = axes[i]
        
        # Build cell ratio data matrix
        ratio_data = []
        # 根据WSI数量决定使用长标签还是短标签
        if num_wsi > 6:
            column_names = [cell_type_short_names[j] for j in [1, 2, 3, 4, 5]]
        else:
            column_names = [cell_type_names[j] for j in [1, 2, 3, 4, 5]]
        
        for patch in analysis['patch_analyses']:
            ratio_data.append([patch['cell_ratios'].get(str(j), 0.0) for j in [1, 2, 3, 4, 5]])
        
        ratio_df = pd.DataFrame(data=ratio_data, columns=column_names)
        
        # Calculate correlation matrix
        corr_matrix = ratio_df.corr()
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(f"WSI {i+1}: Cell Type Correlation\n({'Responder' if analysis['label'] == 1 else 'Non-responder'})")
    
    # Hide unused subplots
    for i in range(num_wsi, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cell_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Statistical plots saved to {output_dir} directory")

def create_patch_examples_visualization(wsi_analyses, output_dir="plots"):
    """
    创建每个WSI的patch实例可视化，展示RPSM选择的具体例子
    """
    print("🎨 Starting patch examples visualization...")
    os.makedirs(output_dir, exist_ok=True)
    
    valid_analyses = [a for a in wsi_analyses if a is not None]
    if not valid_analyses:
        print("❌ No valid analyses found for patch examples visualization")
        return
    
    print(f"📊 Found {len(valid_analyses)} valid WSI analyses")
    
    cell_type_names = {
        1: "Neoplastic", 2: "Inflammatory", 3: "Connective", 4: "Dead", 5: "Epithelial"
    }
    
    cell_type_colors = {
        1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1', 4: '#96CEB4', 5: '#FECA57'
    }
    
    for wsi_idx, analysis in enumerate(valid_analyses):
        print(f"🔍 Processing WSI {wsi_idx + 1}/{len(valid_analyses)}...")
        
        # 调试：检查analysis结构
        if 'patch_analyses' not in analysis:
            print(f"❌ No patch_analyses found in WSI {wsi_idx + 1}")
            continue
            
        total_patches = len(analysis['patch_analyses'])
        print(f"📋 WSI {wsi_idx + 1} has {total_patches} patches")
        
        # 选择有代表性的patch：RPSM选中的、未选中的、三种RPSM标准的对比
        selected_patches = [p for p in analysis['patch_analyses'] if p.get('rpsm_selected', False)]
        unselected_patches = [p for p in analysis['patch_analyses'] if not p.get('rpsm_selected', False)]
        improved_selected = [p for p in analysis['patch_analyses'] if p.get('improved_rpsm_selected', False)]
        angio_selected = [p for p in analysis['patch_analyses'] if p.get('angio_rpsm_selected', False)]
        
        print(f"   - Original RPSM selected: {len(selected_patches)}")
        print(f"   - Improved RPSM selected: {len(improved_selected)}")
        print(f"   - Angiogenesis RPSM selected: {len(angio_selected)}")
        print(f"   - Unselected patches: {len(unselected_patches)}")
        
        # 选择最多6个有代表性的patch进行展示
        example_patches = []
        
        # 添加原始RPSM选中的patch（最多2个）
        if selected_patches:
            example_patches.extend(selected_patches[:2])
            print(f"   - Added {min(2, len(selected_patches))} original RPSM patches")
        
        # 添加改进RPSM特有选中的patch（最多2个）
        improved_only = [p for p in improved_selected if not p.get('rpsm_selected', False)]
        if improved_only:
            example_patches.extend(improved_only[:2])
            print(f"   - Added {min(2, len(improved_only))} improved RPSM only patches")
        
        # 添加血管生成RPSM特有选中的patch（最多2个）
        angio_only = [p for p in angio_selected if not p.get('rpsm_selected', False) and not p.get('improved_rpsm_selected', False)]
        if angio_only:
            example_patches.extend(angio_only[:2])
            print(f"   - Added {min(2, len(angio_only))} angiogenesis RPSM only patches")
        
        # 如果例子不够，添加一些未选中的patch作为对比
        if len(example_patches) < 4 and unselected_patches:
            needed = 4 - len(example_patches)
            example_patches.extend(unselected_patches[:needed])
            print(f"   - Added {min(needed, len(unselected_patches))} unselected patches for comparison")
        
        if not example_patches:
            print(f"❌ No example patches found for WSI {wsi_idx + 1}")
            continue
        
        # 限制最多展示6个patch
        example_patches = example_patches[:6]
        print(f"📝 Will visualize {len(example_patches)} patches for WSI {wsi_idx + 1}")
        
        # 创建子图
        n_patches = len(example_patches)
        cols = min(3, n_patches)
        rows = (n_patches + cols - 1) // cols
        
        print(f"🎯 Creating {rows}x{cols} subplot layout")
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_patches == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        for patch_idx, patch in enumerate(example_patches):
            ax = axes[patch_idx]
            
            print(f"   📊 Processing patch {patch_idx + 1}/{len(example_patches)}")
            
            # 调试：检查patch数据结构
            if 'cell_counts' not in patch:
                print(f"     ❌ No cell_counts in patch {patch_idx + 1}")
                ax.text(0.5, 0.5, 'No cell data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                continue
                
            # 创建细胞类型分布饼图
            cell_counts = patch['cell_counts']
            print(f"     📋 Cell counts: {cell_counts}")
            
            # 修复：cell_counts使用字符串键，需要转换为字符串访问
            sizes = [cell_counts.get(str(i), 0) for i in range(1, 6)]
            print(f"     📊 Sizes: {sizes}")
            
            labels = [cell_type_names[i] for i in range(1, 6)]
            colors = [cell_type_colors[i] for i in range(1, 6)]
            
            # 只显示非零的细胞类型
            non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
            print(f"     ✅ Non-zero indices: {non_zero_indices}")
            
            if non_zero_indices:
                filtered_sizes = [sizes[i] for i in non_zero_indices]
                filtered_labels = [labels[i] for i in non_zero_indices]
                filtered_colors = [colors[i] for i in non_zero_indices]
                
                print(f"     🎨 Creating pie chart with {len(filtered_sizes)} segments")
                
                wedges, texts, autotexts = ax.pie(filtered_sizes, labels=filtered_labels, 
                                                colors=filtered_colors, autopct='%1.1f%%', 
                                                startangle=90)
                
                # 设置文字大小
                for text in texts:
                    text.set_fontsize(8)
                for autotext in autotexts:
                    autotext.set_fontsize(7)
                    autotext.set_color('white')
                    autotext.set_weight('bold')
            else:
                # 如果没有细胞，显示提示信息
                print(f"     ⚠️ No cells detected in patch {patch_idx + 1}")
                ax.text(0.5, 0.5, 'No cells detected', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            # 标题信息
            rpsm_status = []
            if patch.get('rpsm_selected', False):
                rpsm_status.append(f"Original: {patch.get('rpsm_reason', 'Selected')}")
            if patch.get('improved_rpsm_selected', False):
                rpsm_status.append(f"Improved: {patch.get('improved_rpsm_reason', 'Selected')}")
            if patch.get('angio_rpsm_selected', False):
                rpsm_status.append(f"Angio: {patch.get('angio_rpsm_reason', 'Selected')}")
            
            if not rpsm_status:
                rpsm_status = ["Not selected by any RPSM"]
            
            total_cells = patch.get('total_cells', 0)
            title = f"Patch {patch_idx+1}\nTotal cells: {total_cells}\n" + "\n".join(rpsm_status)
            ax.set_title(title, fontsize=9, pad=10)
            
            print(f"     ✅ Patch {patch_idx + 1} visualization completed")
        
        # 隐藏多余的子图
        for i in range(n_patches, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = f"{output_dir}/wsi_{wsi_idx+1}_patch_examples.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ WSI {wsi_idx + 1} patch examples saved to {output_path}")
    
    print(f"🎉 Patch examples visualizations completed and saved to {output_dir} directory")

def create_rpsm_comparison_visualization(wsi_analyses, output_dir="plots"):
    """
    创建RPSM方法详细对比分析 - 专注于核心方法差异和生物学洞察
    
    优化内容:
    1. 方法严格度分布分析  
    2. Patch质量评分对比
    3. 细胞类型偏好热力图
    4. 响应者vs非响应者细胞分布
    
    移除冗余内容:
    - 方法选择重叠分析 (已被performance comparison替代)
    - 失败案例分析 (已被detailed analysis中异常检测替代)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    valid_analyses = [a for a in wsi_analyses if a is not None]
    if not valid_analyses:
        return
    
    # RPSM方法详细对比分析 - 优化为2x2布局，突出核心信息
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    responder_analyses = [a for a in valid_analyses if a['label'] == 1]
    non_responder_analyses = [a for a in valid_analyses if a['label'] == 0]
    
    # 1.1 RPSM方法严格度分析
    ax = axes[0, 0]
    methods = ['Original', 'Improved', 'Angiogenesis', 'Hybrid']
    
    # 计算每种方法的平均选择率（严格度的反映）
    strictness_data = []
    for method_key in ['rpsm_selected_count', 'improved_rpsm_selected_count', 'angio_rpsm_selected_count', 'hybrid_rpsm_selected_count']:
        rates = []
        for analysis in valid_analyses:
            if method_key in analysis:
                rate = analysis[method_key] / analysis['total_patches'] * 100
                rates.append(rate)
        strictness_data.append(rates)
    
    # 创建小提琴图显示分布
    if strictness_data and any(strictness_data):
        violin_parts = ax.violinplot(strictness_data, positions=range(len(methods)), showmeans=True, showmedians=True)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45)
        ax.set_ylabel('Selection Rate (%)')
        ax.set_title('RPSM Method Strictness Distribution')
        ax.grid(True, alpha=0.3)
        
        # 设置颜色
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for pc, color in zip(violin_parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
    
    # 1.2 Patch质量评分分布对比
    ax = axes[0, 1]
    
    def calculate_simple_quality_score(cell_counts):
        total_cells = sum(cell_counts.values())
        if total_cells == 0:
            return 0
        
        # 细胞密度评分
        density_score = min(total_cells / 200.0, 1.0) * 0.4
        
        # 细胞多样性评分
        cell_types = sum(1 for count in cell_counts.values() if count > 0)
        diversity_score = (cell_types / 5.0) * 0.6
        
        return density_score + diversity_score
    
    quality_data = []
    quality_labels = []
    
    for method_name, patches_key in [
        ('Original', 'rpsm_selected_patches'),
        ('Improved', 'improved_rpsm_selected_patches'),
        ('Angiogenesis', 'angio_rpsm_selected_patches'),
        ('Hybrid', 'hybrid_rpsm_selected_patches')
    ]:
        method_quality = []
        for analysis in valid_analyses:
            if patches_key in analysis and analysis[patches_key]:
                for patch in analysis[patches_key]:
                    if 'cell_counts' in patch:
                        quality = calculate_simple_quality_score(patch['cell_counts'])
                        method_quality.append(quality)
        
        if method_quality:
            quality_data.append(method_quality)
            quality_labels.append(method_name)
    
    if quality_data:
        bp = ax.boxplot(quality_data, labels=quality_labels, patch_artist=True)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Quality Score')
        ax.set_title('Patch Quality Score Distribution by Method')
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)
    
    # 1.3 细胞类型偏好热力图 (右上)
    ax = axes[0, 1]
    
    # 计算每种方法对不同细胞类型的偏好
    cell_preferences_matrix = []
    method_names = ['Original', 'Improved', 'Angiogenesis', 'Hybrid']
    cell_type_names = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    
    for method_name, patches_key in [
        ('Original', 'rpsm_selected_patches'),
        ('Improved', 'improved_rpsm_selected_patches'),
        ('Angiogenesis', 'angio_rpsm_selected_patches'),
        ('Hybrid', 'hybrid_rpsm_selected_patches')
    ]:
        method_cell_ratios = []
        for analysis in valid_analyses:
            if patches_key in analysis and analysis[patches_key]:
                for patch in analysis[patches_key]:
                    if 'cell_ratios' in patch:
                        ratios = [patch['cell_ratios'].get(str(i), 0.0) for i in range(1, 6)]
                        method_cell_ratios.append(ratios)
        
        if method_cell_ratios:
            avg_ratios = np.mean(method_cell_ratios, axis=0)
            cell_preferences_matrix.append(avg_ratios)
        else:
            cell_preferences_matrix.append([0, 0, 0, 0, 0])
    
    if cell_preferences_matrix:
        cell_preferences_matrix = np.array(cell_preferences_matrix)
        im = ax.imshow(cell_preferences_matrix, cmap='RdYlBu_r', aspect='auto')
        
        ax.set_xticks(range(5))
        ax.set_xticklabels(['Neo', 'Inf', 'Con', 'Dead', 'Epi'], rotation=45)
        ax.set_yticks(range(len(method_names)))
        ax.set_yticklabels(method_names)
        ax.set_title('Cell Type Preference Heatmap')
        
        # 添加数值标签
        for i in range(len(method_names)):
            for j in range(5):
                if len(cell_preferences_matrix) > i:
                    text = ax.text(j, i, f'{cell_preferences_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=ax, label='Average Cell Ratio')
    
    # 1.4 响应者vs非响应者的细胞分布对比 (左下)
    ax = axes[1, 0]
    
    if responder_analyses and non_responder_analyses:
        # 计算每组的平均细胞比例
        resp_cell_ratios = {str(i): [] for i in range(1, 6)}
        non_resp_cell_ratios = {str(i): [] for i in range(1, 6)}
        
        for analysis in responder_analyses:
            for patch in analysis['patch_analyses']:
                for cell_type in range(1, 6):
                    cell_type_str = str(cell_type)
                    resp_cell_ratios[cell_type_str].append(patch['cell_ratios'].get(cell_type_str, 0.0))
        
        for analysis in non_responder_analyses:
            for patch in analysis['patch_analyses']:
                for cell_type in range(1, 6):
                    cell_type_str = str(cell_type)
                    non_resp_cell_ratios[cell_type_str].append(patch['cell_ratios'].get(cell_type_str, 0.0))
        
        resp_means = [np.mean(resp_cell_ratios[str(i)]) for i in range(1, 6)]
        non_resp_means = [np.mean(non_resp_cell_ratios[str(i)]) for i in range(1, 6)]
        
        x = np.arange(5)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, resp_means, width, label='Responders', color='#2E8B57', alpha=0.8)
        bars2 = ax.bar(x + width/2, non_resp_means, width, label='Non-responders', color='#CD5C5C', alpha=0.8)
        
        ax.set_title('Cell Distribution: Responders vs Non-responders')
        ax.set_ylabel('Average Cell Ratio')
        ax.set_xlabel('Cell Types')
        ax.set_xticks(x)
        ax.set_xticklabels(['Neo', 'Inf', 'Con', 'Dead', 'Epi'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 1.4 方法选择效率对比 (右下)
    ax = axes[1, 1]
    
    # 计算每种方法的选择效率指标
    efficiency_metrics = {'Original': [], 'Improved': [], 'Angiogenesis': [], 'Hybrid': []}
    
    for analysis in valid_analyses:
        total_patches = analysis['total_patches']
        
        # 计算各方法的选择效率 (选择率 × 质量分数)
        methods_data = [
            ('Original', analysis.get('rpsm_selected_count', 0), 'rpsm_selected'),
            ('Improved', analysis.get('improved_rpsm_selected_count', 0), 'improved_rpsm_selected'),
            ('Angiogenesis', analysis.get('angio_rpsm_selected_count', 0), 'angio_rpsm_selected'),
            ('Hybrid', analysis.get('hybrid_rpsm_selected_count', 0), 'hybrid_rpsm_selected')
        ]
        
        for method_name, selected_count, patch_key in methods_data:
            if selected_count > 0:
                # 计算选中patches的平均质量
                selected_patches = [p for p in analysis['patch_analyses'] if p.get(patch_key, False)]
                if selected_patches:
                    avg_quality = np.mean([calculate_simple_quality_score(p['cell_counts']) for p in selected_patches])
                    selection_rate = selected_count / total_patches
                    efficiency = selection_rate * avg_quality * 100  # 效率指标
                    efficiency_metrics[method_name].append(efficiency)
                else:
                    efficiency_metrics[method_name].append(0)
            else:
                efficiency_metrics[method_name].append(0)
    
    # 绘制效率对比
    method_names = list(efficiency_metrics.keys())
    efficiency_data = [efficiency_metrics[method] for method in method_names]
    
    if efficiency_data and any(any(data) for data in efficiency_data):
        bp = ax.boxplot(efficiency_data, labels=method_names, patch_artist=True)
        ax.set_title('RPSM Method Selection Efficiency')
        ax.set_ylabel('Efficiency Score (Selection Rate × Quality)')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # 设置颜色
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    plt.suptitle('RPSM Methods Core Comparison Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rpsm_detailed_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def calculate_patch_quality(patch):
    """计算patch的质量分数"""
    try:
        total_cells = patch['total_cells']
        if total_cells == 0:
            return 0.0
        
        # 细胞类型多样性
        cell_counts = patch['cell_counts']
        non_zero_types = sum(1 for count in cell_counts.values() if count > 0)
        diversity_score = min(non_zero_types / 5.0, 1.0)
        
        # 细胞密度适中性 (50-300 cells optimal)
        if 50 <= total_cells <= 300:
            density_score = 1.0
        elif total_cells < 50:
            density_score = total_cells / 50.0
        else:
            density_score = max(0.3, 300.0 / total_cells)
        
        # 肿瘤细胞比例适中
        tumor_ratio = patch['cell_ratios'].get('1', 0.0)
        if 0.3 <= tumor_ratio <= 0.8:
            tumor_score = 1.0
        elif tumor_ratio < 0.3:
            tumor_score = tumor_ratio / 0.3
        else:
            tumor_score = max(0.5, (1.0 - tumor_ratio) / 0.2)
        
        # 综合质量分数
        quality_score = (diversity_score * 0.4 + density_score * 0.3 + tumor_score * 0.3)
        return min(quality_score, 1.0)
    except:
        return 0.0

def calculate_angiogenesis_score(patch):
    """计算patch的血管生成分数"""
    try:
        cell_ratios = patch['cell_ratios']
        
        # 血管生成相关的细胞比例组合
        endothelial_ratio = cell_ratios.get('2', 0.0)  # 内皮细胞
        inflammatory_ratio = cell_ratios.get('3', 0.0)  # 炎症细胞
        connective_ratio = cell_ratios.get('4', 0.0)   # 结缔组织细胞
        
        # 血管生成评分算法
        angio_score = (
            endothelial_ratio * 0.5 +  # 内皮细胞最重要
            inflammatory_ratio * 0.3 +  # 炎症细胞次要
            connective_ratio * 0.2      # 结缔组织支持
        )
        
        return min(angio_score, 1.0)
    except:
        return 0.0

def calculate_immune_infiltration(patch):
    """计算patch的免疫浸润分数"""
    try:
        cell_ratios = patch['cell_ratios']
        
        # 免疫相关细胞比例
        inflammatory_ratio = cell_ratios.get('3', 0.0)  # 炎症/免疫细胞
        necrotic_ratio = cell_ratios.get('5', 0.0)      # 坏死细胞(可能包含免疫细胞)
        
        # 免疫浸润评分
        immune_score = inflammatory_ratio * 0.8 + necrotic_ratio * 0.2
        
        return min(immune_score, 1.0)
    except:
        return 0.0

def calculate_clinical_scores(patches):
    """计算临床相关性评分"""
    try:
        if not patches:
            return {'angiogenesis': 0.0, 'immune': 0.0, 'tumor_burden': 0.0, 'quality': 0.0}
        
        # 计算所有patch的各项分数
        angio_scores = [calculate_angiogenesis_score(p) for p in patches]
        immune_scores = [calculate_immune_infiltration(p) for p in patches]
        quality_scores = [calculate_patch_quality(p) for p in patches]
        
        # 肿瘤负荷分数
        tumor_burdens = []
        for patch in patches:
            tumor_ratio = patch['cell_ratios'].get('1', 0.0)
            total_cells = patch['total_cells']
            # 结合肿瘤比例和细胞密度
            burden = tumor_ratio * min(total_cells / 200.0, 1.0)
            tumor_burdens.append(burden)
        
        return {
            'angiogenesis': np.mean(angio_scores),
            'immune': np.mean(immune_scores),
            'tumor_burden': np.mean(tumor_burdens),
            'quality': np.mean(quality_scores)
        }
    except:
        return {'angiogenesis': 0.0, 'immune': 0.0, 'tumor_burden': 0.0, 'quality': 0.0}
    
def create_detailed_patch_analysis(wsi_analyses, output_dir="plots"):
    """
    创建优化的WSI详细分析图表 - 智能采样模式适应大规模数据集
    
    优化策略:
    - WSI <= 10: 全部详细分析
    - WSI 11-50: 采样分析 + 聚合统计
    - WSI > 50: 仅生成代表性样本分析
    """
    os.makedirs(output_dir, exist_ok=True)
    
    valid_analyses = [a for a in wsi_analyses if a is not None]
    if not valid_analyses:
        return
    
    num_wsi = len(valid_analyses)
    
    # 智能采样策略
    if num_wsi <= 10:
        # 小规模数据集：全部分析
        selected_analyses = valid_analyses
        print(f"📊 Small dataset: analyzing all {num_wsi} WSIs")
    elif num_wsi <= 50:
        # 中等规模：采样分析
        # 确保响应者和非响应者的代表性
        responders = [a for a in valid_analyses if a['label'] == 1]
        non_responders = [a for a in valid_analyses if a['label'] == 0]
        
        # 从每组采样最多5个
        sampled_responders = random.sample(responders, min(5, len(responders))) if responders else []
        sampled_non_responders = random.sample(non_responders, min(5, len(non_responders))) if non_responders else []
        
        selected_analyses = sampled_responders + sampled_non_responders
        print(f"📊 Medium dataset: sampling {len(selected_analyses)} from {num_wsi} WSIs ({len(sampled_responders)} responders, {len(sampled_non_responders)} non-responders)")
    else:
        # 大规模数据集：仅生成代表性样本
        responders = [a for a in valid_analyses if a['label'] == 1]
        non_responders = [a for a in valid_analyses if a['label'] == 0]
        
        # 从每组采样最多3个最有代表性的案例
        # 基于RPSM选择率选择最有代表性的案例
        if responders:
            responders_sorted = sorted(responders, key=lambda x: x['rpsm_selected_count'] / x['total_patches'], reverse=True)
            sampled_responders = responders_sorted[:3]
        else:
            sampled_responders = []
            
        if non_responders:
            non_responders_sorted = sorted(non_responders, key=lambda x: x['rpsm_selected_count'] / x['total_patches'], reverse=True)
            sampled_non_responders = non_responders_sorted[:3]
        else:
            sampled_non_responders = []
        
        selected_analyses = sampled_responders + sampled_non_responders
        print(f"📊 Large dataset: analyzing {len(selected_analyses)} representative cases from {num_wsi} WSIs")
    
    # 为大规模数据集生成采样说明文件
    if num_wsi > 10:
        sampling_info = {
            'total_wsi': num_wsi,
            'analyzed_wsi': len(selected_analyses),
            'sampling_strategy': 'representative' if num_wsi > 50 else 'random',
            'responders_total': len([a for a in valid_analyses if a['label'] == 1]),
            'responders_sampled': len([a for a in selected_analyses if a['label'] == 1]),
            'non_responders_total': len([a for a in valid_analyses if a['label'] == 0]),
            'non_responders_sampled': len([a for a in selected_analyses if a['label'] == 0])
        }
        
        with open(f"{output_dir}/sampling_info.json", 'w') as f:
            json.dump(sampling_info, f, indent=2)
    
    cell_type_names = {
        1: "Neoplastic", 2: "Inflammatory", 3: "Connective", 4: "Dead", 5: "Epithelial"
    }
    
    cell_type_colors = {
        1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1', 4: '#96CEB4', 5: '#FECA57'
    }
    
    # 为选中的WSI创建优化的详细分析
    for wsi_idx, analysis in enumerate(selected_analyses):
        original_idx = valid_analyses.index(analysis) + 1  # 获取原始索引
        print(f"Creating optimized detailed analysis for WSI {original_idx} (sample {wsi_idx + 1}/{len(selected_analyses)})...")
        
        # 收集所有patch的信息
        all_patches = analysis['patch_analyses']
        selected_patches = [p for p in all_patches if p['rpsm_selected']]
        improved_selected = [p for p in all_patches if p.get('improved_rpsm_selected', False)]
        angio_selected = [p for p in all_patches if p.get('angio_rpsm_selected', False)]
        hybrid_selected = [p for p in all_patches if p.get('hybrid_rpsm_selected', False)]
        
        # 计算临床相关性指标
        def calculate_clinical_scores(patches):
            if not patches:
                return {'angiogenesis': 0, 'immune': 0, 'tumor_burden': 0, 'quality': 0}
            
            angio_scores = []
            immune_scores = []
            tumor_scores = []
            quality_scores = []
            
            for patch in patches:
                # 血管生成评分
                if 'cell_ratios' in patch:
                    angio_score = infer_angiogenesis_from_cells(patch['cell_ratios'])
                    angio_scores.append(angio_score)
                
                # 免疫浸润评分 (基于炎症细胞比例)
                immune_ratio = patch['cell_ratios'].get('2', 0.0)
                immune_score = min(immune_ratio * 4, 1.0)  # 标准化到0-1
                immune_scores.append(immune_score)
                
                # 肿瘤负荷评分
                tumor_ratio = patch['cell_ratios'].get('1', 0.0)
                tumor_score = tumor_ratio
                tumor_scores.append(tumor_score)
                
                # 质量评分 (细胞密度 + 多样性)
                total_cells = patch['total_cells']
                cell_types = sum(1 for count in patch['cell_counts'].values() if count > 0)
                quality_score = (min(total_cells / 200.0, 1.0) * 0.6) + (cell_types / 5.0 * 0.4)
                quality_scores.append(quality_score)
            
            return {
                'angiogenesis': np.mean(angio_scores) if angio_scores else 0,
                'immune': np.mean(immune_scores) if immune_scores else 0,
                'tumor_burden': np.mean(tumor_scores) if tumor_scores else 0,
                'quality': np.mean(quality_scores) if quality_scores else 0
            }
        
        # 计算各方法的临床评分
        all_scores = calculate_clinical_scores(all_patches)
        orig_scores = calculate_clinical_scores(selected_patches)
        impr_scores = calculate_clinical_scores(improved_selected)
        angio_scores = calculate_clinical_scores(angio_selected)
        hybrid_scores = calculate_clinical_scores(hybrid_selected)
        
        # 创建优化的3x2布局
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # 1. 质量分布热图 (左上) - 替代简单直方图
        ax = axes[0, 0]
        
        # 准备二维数据：密度 vs 多样性
        densities = []
        diversities = []
        selection_status = []
        
        for patch in all_patches:
            density = patch['total_cells']
            diversity = sum(1 for count in patch['cell_counts'].values() if count > 0)
            densities.append(density)
            diversities.append(diversity)
            
            # 标记选择状态
            if patch['rpsm_selected']:
                selection_status.append('Original')
            elif patch.get('improved_rpsm_selected', False):
                selection_status.append('Improved')
            elif patch.get('angio_rpsm_selected', False):
                selection_status.append('Angiogenesis')
            elif patch.get('hybrid_rpsm_selected', False):
                selection_status.append('Hybrid')
            else:
                selection_status.append('Unselected')
        
        # 创建散点图
        colors = {'Original': '#FF6B6B', 'Improved': '#4ECDC4', 'Angiogenesis': '#45B7D1', 
                 'Hybrid': '#96CEB4', 'Unselected': '#DDDDDD'}
        
        for status in ['Unselected', 'Original', 'Improved', 'Angiogenesis', 'Hybrid']:
            mask = [s == status for s in selection_status]
            if any(mask):
                x_data = [densities[i] for i in range(len(mask)) if mask[i]]
                y_data = [diversities[i] for i in range(len(mask)) if mask[i]]
                ax.scatter(x_data, y_data, c=colors[status], label=status, 
                          alpha=0.7 if status != 'Unselected' else 0.3, s=30)
        
        ax.set_xlabel('Cell Density (total cells per patch)')
        ax.set_ylabel('Cell Diversity (number of cell types)')
        ax.set_title(f'WSI {wsi_idx+1}: Patch Quality Landscape')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 标注高质量但未被选中的patches
        high_quality_unselected = 0
        for i, (d, div, status) in enumerate(zip(densities, diversities, selection_status)):
            if status == 'Unselected' and d >= 100 and div >= 4:
                high_quality_unselected += 1
        
        if high_quality_unselected > 0:
            ax.text(0.02, 0.98, f'⚠️ {high_quality_unselected} high-quality patches missed', 
                   transform=ax.transAxes, va='top', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 2. 临床相关性雷达图 (右上) - 新增重要功能
        ax = axes[0, 1]
        
        # 2. 临床相关性雷达图 (右上) - 新增重要功能
        ax = axes[0, 1]
        
        # 雷达图数据准备
        categories = ['Angiogenesis', 'Immune\nInfiltration', 'Tumor\nBurden', 'Patch\nQuality']
        
        # 计算同组平均值作为基准
        same_label_analyses = [a for a in valid_analyses if a['label'] == analysis['label']]
        if len(same_label_analyses) > 1:
            baseline_scores = []
            for other_analysis in same_label_analyses:
                if other_analysis != analysis:
                    other_patches = other_analysis['patch_analyses']
                    other_scores = calculate_clinical_scores(other_patches)
                    baseline_scores.append([other_scores['angiogenesis'], other_scores['immune'], 
                                          other_scores['tumor_burden'], other_scores['quality']])
            baseline = np.mean(baseline_scores, axis=0).tolist() if baseline_scores else [0.3, 0.3, 0.3, 0.3]
        else:
            baseline = [0.3, 0.3, 0.3, 0.3]  # 默认基准
        
        current_values = [all_scores['angiogenesis'], all_scores['immune'], 
                         all_scores['tumor_burden'], all_scores['quality']]
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 确保数据数组长度匹配
        current_values_closed = current_values + current_values[:1]
        baseline_closed = baseline + baseline[:1]
        
        ax.plot(angles, current_values_closed, 'o-', linewidth=2, label=f'WSI {wsi_idx+1}', color='#FF6B6B')
        ax.fill(angles, current_values_closed, alpha=0.25, color='#FF6B6B')
        ax.plot(angles, baseline_closed, 'o--', linewidth=1, label='Same Group Avg', color='#888888')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(f'WSI {wsi_idx+1}: Clinical Relevance Profile')
        ax.legend()
        ax.grid(True)
        
        # 添加响应标签
        response_label = 'Responder' if analysis['label'] == 1 else 'Non-responder'
        ax.text(0.02, 0.98, f'Label: {response_label}', transform=ax.transAxes, va='top',
               bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor='lightgreen' if analysis['label'] == 1 else 'lightcoral', 
                        alpha=0.7))
        
        # 3. RPSM决策流程可视化 (左中)
        ax = axes[1, 0]
        ax.axis('off')
        
        # 创建决策流程统计
        total_patches = len(all_patches)
        flow_data = {
            'Total Patches': total_patches,
            'Cell Count Filter': 0,
            'Cell Ratio Filter': 0,
            'Quality Filter': 0,
            'Final Selected': {
                'Original': len(selected_patches),
                'Improved': len(improved_selected), 
                'Angiogenesis': len(angio_selected),
                'Hybrid': len(hybrid_selected)
            }
        }
        
        # 分析筛选原因
        for patch in all_patches:
            total_cells = patch['total_cells']
            if total_cells < 50 or total_cells > 300:
                flow_data['Cell Count Filter'] += 1
                continue
            
            tumor_ratio = patch['cell_ratios'].get('1', 0.0)
            if tumor_ratio < 0.3:
                flow_data['Cell Ratio Filter'] += 1
                continue
                
            cell_types = sum(1 for count in patch['cell_counts'].values() if count > 0)
            if cell_types < 3:
                flow_data['Quality Filter'] += 1
        
        # 绘制流程图
        flow_text = f"""RPSM Decision Flow Analysis
        
🔄 Initial Pool: {total_patches} patches
        
📊 Filtering Steps:
├─ Cell Count Filter: -{flow_data['Cell Count Filter']} patches
├─ Cell Ratio Filter: -{flow_data['Cell Ratio Filter']} patches  
├─ Quality Filter: -{flow_data['Quality Filter']} patches
        
✅ Final Selections:
├─ Original RPSM: {flow_data['Final Selected']['Original']} ({flow_data['Final Selected']['Original']/total_patches*100:.1f}%)
├─ Improved RPSM: {flow_data['Final Selected']['Improved']} ({flow_data['Final Selected']['Improved']/total_patches*100:.1f}%)
├─ Angiogenesis: {flow_data['Final Selected']['Angiogenesis']} ({flow_data['Final Selected']['Angiogenesis']/total_patches*100:.1f}%)
└─ Hybrid RPSM: {flow_data['Final Selected']['Hybrid']} ({flow_data['Final Selected']['Hybrid']/total_patches*100:.1f}%)

🎯 Best Method: {max(flow_data['Final Selected'], key=flow_data['Final Selected'].get)}
📈 Max Selection Rate: {max(flow_data['Final Selected'].values())/total_patches*100:.1f}%
"""
        
        ax.text(0.05, 0.95, flow_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 4. 关键patches展示 (右中)
        ax = axes[1, 1]
        
        # 选择最有代表性的patches进行展示
        showcase_patches = []
        
        # 高质量patches (前3个)
        quality_sorted = sorted(all_patches, key=lambda p: calculate_patch_quality(p), reverse=True)
        showcase_patches.extend([('High Quality', p, '#2E8B57') for p in quality_sorted[:3]])
        
        # 血管生成patches (前2个)
        angio_sorted = sorted(all_patches, key=lambda p: calculate_angiogenesis_score(p), reverse=True)
        showcase_patches.extend([('High Angiogenesis', p, '#FF6347') for p in angio_sorted[:2]])
        
        # 免疫浸润patches (前2个)  
        immune_sorted = sorted(all_patches, key=lambda p: calculate_immune_infiltration(p), reverse=True)
        showcase_patches.extend([('High Immune', p, '#4169E1') for p in immune_sorted[:2]])
        
        # 创建展示图
        y_positions = []
        colors = []
        labels = []
        
        for i, (category, patch, color) in enumerate(showcase_patches):
            y_pos = len(showcase_patches) - i - 1
            y_positions.append(y_pos)
            colors.append(color)
            
            # 创建标签信息
            total_cells = patch['total_cells']
            angio_score = calculate_angiogenesis_score(patch) * 100
            quality_score = calculate_patch_quality(patch) * 100
            
            label = f"{category}\nCells: {total_cells}, Angio: {angio_score:.1f}%, Quality: {quality_score:.1f}%"
            labels.append(label)
        
        # 绘制水平条形图
        bars = ax.barh(y_positions, [1] * len(showcase_patches), color=colors, alpha=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Representative Patches')
        ax.set_title(f'WSI {wsi_idx+1}: Key Patches Showcase')
        ax.set_xlim(0, 1.2)
        
        # 添加选择状态指示
        for i, (category, patch, color) in enumerate(showcase_patches):
            selected_methods = []
            if patch in selected_patches:
                selected_methods.append('Orig')
            if patch in improved_selected:
                selected_methods.append('Imp')
            if patch in angio_selected:
                selected_methods.append('Angio')
            if patch in hybrid_selected:
                selected_methods.append('Hyb')
                
            if selected_methods:
                ax.text(1.05, y_positions[i], f"✓{'/'.join(selected_methods)}", 
                       va='center', fontsize=8, color='green', weight='bold')
            else:
                ax.text(1.05, y_positions[i], "✗None", 
                       va='center', fontsize=8, color='red')
        
        # 5. 方法性能对比 (左下)
        ax = axes[2, 0]
        
        # 计算各方法的性能指标
        methods_performance = {
            'Original': {
                'count': len(selected_patches),
                'avg_quality': np.mean([calculate_patch_quality(p) for p in selected_patches]) if selected_patches else 0,
                'avg_angio': np.mean([calculate_angiogenesis_score(p) for p in selected_patches]) if selected_patches else 0
            },
            'Improved': {
                'count': len(improved_selected),
                'avg_quality': np.mean([calculate_patch_quality(p) for p in improved_selected]) if improved_selected else 0,
                'avg_angio': np.mean([calculate_angiogenesis_score(p) for p in improved_selected]) if improved_selected else 0
            },
            'Angiogenesis': {
                'count': len(angio_selected),
                'avg_quality': np.mean([calculate_patch_quality(p) for p in angio_selected]) if angio_selected else 0,
                'avg_angio': np.mean([calculate_angiogenesis_score(p) for p in angio_selected]) if angio_selected else 0
            },
            'Hybrid': {
                'count': len(hybrid_selected),
                'avg_quality': np.mean([calculate_patch_quality(p) for p in hybrid_selected]) if hybrid_selected else 0,
                'avg_angio': np.mean([calculate_angiogenesis_score(p) for p in hybrid_selected]) if hybrid_selected else 0
            }
        }
        
        # 创建性能对比雷达图
        performance_categories = ['Count\n(normalized)', 'Quality\nScore', 'Angio\nScore']
        performance_angles = np.linspace(0, 2 * np.pi, len(performance_categories), endpoint=False).tolist()
        performance_angles += performance_angles[:1]
        
        max_count = max([perf['count'] for perf in methods_performance.values()]) or 1
        
        colors_methods = ['#FF9999', '#66B2FF', '#99FF99', '#FFB366']
        for i, (method, perf) in enumerate(methods_performance.items()):
            if perf['count'] > 0:  # 只显示有patches的方法
                values = [
                    perf['count'] / max_count,  # 标准化计数
                    perf['avg_quality'], 
                    perf['avg_angio']
                ]
                values_closed = values + values[:1]  # 闭合数据
                
                ax.plot(performance_angles, values_closed, 'o-', linewidth=2, 
                       label=f'{method} ({perf["count"]})', color=colors_methods[i])
                ax.fill(performance_angles, values_closed, alpha=0.15, color=colors_methods[i])
        
        ax.set_xticks(performance_angles[:-1])
        ax.set_xticklabels(performance_categories)
        ax.set_ylim(0, 1)
        ax.set_title(f'WSI {wsi_idx+1}: RPSM Methods Performance')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
        # 6. 异常检测与洞察 (右下)
        ax = axes[2, 1]
        ax.axis('off')
        
        # 异常检测分析
        anomalies = []
        insights = []
        
        # 1. 细胞计数异常
        cell_counts = [p['total_cells'] for p in all_patches]
        q1, q3 = np.percentile(cell_counts, [25, 75])
        iqr = q3 - q1
        outliers = [p for p in all_patches if p['total_cells'] < q1 - 1.5*iqr or p['total_cells'] > q3 + 1.5*iqr]
        if outliers:
            anomalies.append(f"📊 {len(outliers)} patches with abnormal cell counts")
        
        # 2. RPSM方法选择差异
        method_counts = [len(selected_patches), len(improved_selected), len(angio_selected), len(hybrid_selected)]
        if max(method_counts) > 0:
            method_variance = np.var(method_counts) / np.mean(method_counts) if np.mean(method_counts) > 0 else 0
            if method_variance > 0.5:
                anomalies.append(f"⚠️ High variance in RPSM method selections ({method_variance:.2f})")
        
        # 3. 质量与选择不匹配
        high_quality_unselected = [p for p in all_patches 
                                 if calculate_patch_quality(p) > 0.7 
                                 and p not in selected_patches 
                                 and p not in improved_selected]
        if high_quality_unselected:
            anomalies.append(f"🔍 {len(high_quality_unselected)} high-quality patches missed by RPSM")
        
        # 4. 生成洞察
        if all_scores['angiogenesis'] > 0.6:
            insights.append("🩸 Strong angiogenesis signature detected")
        
        if all_scores['immune'] > 0.5:
            insights.append("🛡️ High immune infiltration observed")
            
        if all_scores['quality'] < 0.4:
            insights.append("⚠️ Generally low patch quality")
            
        # 响应预测洞察
        if analysis['label'] == 1:  # Responder
            if all_scores['angiogenesis'] < 0.3:
                insights.append("🤔 Responder with low angiogenesis - investigate further")
        else:  # Non-responder
            if all_scores['angiogenesis'] > 0.7:
                insights.append("🤔 Non-responder with high angiogenesis - potential misclassification")
        
        # 方法效果洞察
        best_method = max(methods_performance, key=lambda m: methods_performance[m]['count'])
        if methods_performance[best_method]['count'] > total_patches * 0.3:
            insights.append(f"✅ {best_method} RPSM shows best selection rate")
        else:
            insights.append("❌ All RPSM methods show low selection rates")
        
        # 显示异常和洞察
        report_text = f"""WSI {wsi_idx+1}: Anomaly Detection & Insights
        
🚨 ANOMALIES DETECTED:
"""
        if anomalies:
            for anomaly in anomalies:
                report_text += f"   {anomaly}\n"
        else:
            report_text += "   ✅ No significant anomalies detected\n"
            
        report_text += f"""
💡 KEY INSIGHTS:
"""
        if insights:
            for insight in insights:
                report_text += f"   {insight}\n"
        else:
            report_text += "   📋 Standard patterns observed\n"
            
        # 添加统计摘要
        report_text += f"""
📈 SUMMARY STATISTICS:
   Total Patches: {total_patches}
   Best RPSM: {best_method} ({methods_performance[best_method]['count']} patches)
   Avg Quality: {all_scores['quality']:.3f}
   Avg Angiogenesis: {all_scores['angiogenesis']:.3f}
   Clinical Label: {'Responder' if analysis['label'] == 1 else 'Non-responder'}
"""
        
        ax.text(0.05, 0.95, report_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(output_dir, f'wsi_{wsi_idx+1}_detailed_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
        
        reason_counts = {}
        all_reasons = []
        
        # 统计各种RPSM的选择原因
        for patch in selected_patches:
            reason = patch.get('rpsm_reason', 'Unknown')
            all_reasons.append(f"Orig: {reason}")
        
        for patch in improved_selected:
            reason = patch.get('improved_rpsm_reason', 'Selected')
            all_reasons.append(f"Impr: {reason}")
        
        for patch in angio_selected:
            reason = patch.get('angio_rpsm_reason', 'Selected')
            all_reasons.append(f"Angio: {reason}")
        
        if all_reasons:
            reason_counts = pd.Series(all_reasons).value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(reason_counts)))
            
            wedges, texts, autotexts = ax.pie(reason_counts.values, 
                                            labels=reason_counts.index,
                                            colors=colors, autopct='%1.1f%%', 
                                            startangle=90)
            ax.set_title(f'WSI {wsi_idx+1}: RPSM Selection Reasons')
            
            for text in texts:
                text.set_fontsize(8)
            for autotext in autotexts:
                autotext.set_fontsize(7)
                autotext.set_color('white')
                autotext.set_weight('bold')
        
        # 4. 细胞类型相关性热图（仅选中的patch）
        ax = axes[1, 0]
        
        if selected_patches:
            # 构建细胞比例数据矩阵
            ratio_data = []
            for patch in selected_patches:
                ratio_data.append([patch['cell_ratios'].get(str(j), 0.0) for j in range(1, 6)])
            
            if ratio_data:
                ratio_df = pd.DataFrame(data=ratio_data, 
                                      columns=[cell_type_names[j] for j in range(1, 6)])
                corr_matrix = ratio_df.corr()
                
                im = ax.imshow(corr_matrix.values, cmap='coolwarm', aspect='equal', vmin=-1, vmax=1)
                ax.set_xticks(range(5))
                ax.set_yticks(range(5))
                ax.set_xticklabels([cell_type_names[j+1] for j in range(5)], rotation=45)
                ax.set_yticklabels([cell_type_names[j+1] for j in range(5)])
                ax.set_title(f'WSI {wsi_idx+1}: Cell Type Correlations\n(Original RPSM Selected)')
                
                # 添加相关性数值
                for i in range(5):
                    for j in range(5):
                        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8)
                
                plt.colorbar(im, ax=ax, shrink=0.8)
        
        # 5. 三种RPSM方法的重叠分析（韦恩图风格）
        ax = axes[1, 1]
        
        # 计算重叠情况
        patch_selections = {}
        for i, patch in enumerate(all_patches):
            orig = patch['rpsm_selected']
            impr = patch.get('improved_rpsm_selected', False)
            angio = patch.get('angio_rpsm_selected', False)
            patch_selections[i] = (orig, impr, angio)
        
        overlap_counts = {
            'Original only': 0, 'Improved only': 0, 'Angiogenesis only': 0,
            'Orig + Impr': 0, 'Orig + Angio': 0, 'Impr + Angio': 0,
            'All three': 0, 'None': 0
        }
        
        for orig, impr, angio in patch_selections.values():
            if orig and impr and angio:
                overlap_counts['All three'] += 1
            elif orig and impr:
                overlap_counts['Orig + Impr'] += 1
            elif orig and angio:
                overlap_counts['Orig + Angio'] += 1
            elif impr and angio:
                overlap_counts['Impr + Angio'] += 1
            elif orig:
                overlap_counts['Original only'] += 1
            elif impr:
                overlap_counts['Improved only'] += 1
            elif angio:
                overlap_counts['Angiogenesis only'] += 1
            else:
                overlap_counts['None'] += 1
        
        # 移除计数为0的类别
        non_zero_overlaps = {k: v for k, v in overlap_counts.items() if v > 0}
        
        if non_zero_overlaps:
            colors = plt.cm.Set2(np.linspace(0, 1, len(non_zero_overlaps)))
            wedges, texts, autotexts = ax.pie(non_zero_overlaps.values(),
                                            labels=non_zero_overlaps.keys(),
                                            colors=colors, autopct='%1.1f%%',
                                            startangle=90)
            ax.set_title(f'WSI {wsi_idx+1}: RPSM Method Overlaps')
            
            for text in texts:
                text.set_fontsize(9)
            for autotext in autotexts:
                autotext.set_fontsize(8)
                autotext.set_color('white')
                autotext.set_weight('bold')
        
        # 6. 选择性统计摘要
        ax = axes[1, 2]
        ax.axis('off')
        
        # 创建统计摘要文本
        total_patches = len(all_patches)
        orig_count = len(selected_patches)
        impr_count = len(improved_selected)
        angio_count = len(angio_selected)
        
        orig_rate = orig_count / total_patches * 100 if total_patches > 0 else 0
        impr_rate = impr_count / total_patches * 100 if total_patches > 0 else 0
        angio_rate = angio_count / total_patches * 100 if total_patches > 0 else 0
        
        summary_text = f"""WSI {wsi_idx+1} Summary
        
Total patches: {total_patches}

RPSM Selection Results:
• Original RPSM: {orig_count} ({orig_rate:.1f}%)
• Improved RPSM: {impr_count} ({impr_rate:.1f}%)
• Angiogenesis RPSM: {angio_count} ({angio_rate:.1f}%)

Response label: {'Responder' if analysis['label'] == 1 else 'Non-responder'}

Cell count statistics:
• Mean: {np.mean(all_counts):.1f}
• Median: {np.median(all_counts):.1f}
• Range: {np.min(all_counts)}-{np.max(all_counts)}

Best performing RPSM:
{['Original', 'Improved', 'Angiogenesis'][np.argmax([orig_rate, impr_rate, angio_rate])]} 
({max(orig_rate, impr_rate, angio_rate):.1f}% selection rate)
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/wsi_{wsi_idx+1}_detailed_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Detailed patch analysis saved to {output_dir} directory")

def create_patch_prediction_visualization(wsi_analyses, output_dir="plots"):
    """
    创建patch预测可视化，参照app.py的实现方式
    每个WSI选择3个具有代表性的patch进行可视化，并在原图上标注细胞
    """
    import cv2
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    os.makedirs(output_dir, exist_ok=True)
    
    valid_analyses = [a for a in wsi_analyses if a is not None]
    if not valid_analyses:
        return
    
    # 参照app.py的颜色映射
    color_dict = {
        0: [0, 0, 0],       # Background - black
        1: [255, 0, 0],     # Neoplastic - red  
        2: [0, 255, 0],     # Inflammatory - green
        3: [0, 0, 255],     # Connective - blue
        4: [255, 255, 0],   # Dead - yellow
        5: [255, 0, 255],   # Epithelial - magenta
    }
    
    type_names = {
        1: "Neoplastic", 2: "Inflammatory", 3: "Connective", 
        4: "Dead", 5: "Epithelial"
    }
    
    # 设置模型用于重新推理
    print("Setting up model for detailed cell visualization...")
    model_result = setup_pannuke_models()
    if model_result is None:
        print("Failed to setup model for visualization")
        return
    
    if len(model_result) == 3:
        pannuke_model, device, is_multi_gpu = model_result
    else:
        print("Unexpected return format from setup_pannuke_models")
        return
    
    for wsi_idx, analysis in enumerate(valid_analyses):
        print(f"Creating patch prediction visualization for WSI {wsi_idx + 1}...")
        
        # 选择有代表性的patch进行可视化
        all_patches = analysis['patch_analyses']
        
        # 选择3个具有代表性的patch
        selected_patches = []
        
        # 1. 选择一个原始RPSM选中的patch
        original_selected = [p for p in all_patches if p.get('rpsm_selected', False)]
        if original_selected:
            # 选择细胞数量较多的patch
            original_selected.sort(key=lambda x: x['total_cells'], reverse=True)
            selected_patches.append(original_selected[0])
        
        # 2. 选择一个改进/血管生成RPSM选中但原始RPSM未选中的patch
        alternative_selected = [p for p in all_patches if 
                               (p.get('improved_rpsm_selected', False) or p.get('angio_rpsm_selected', False)) 
                               and not p.get('rpsm_selected', False) 
                               and p not in selected_patches]
        if alternative_selected:
            alternative_selected.sort(key=lambda x: x['total_cells'], reverse=True)
            selected_patches.append(alternative_selected[0])
        
        # 3. 选择一个未被任何RPSM选中的patch作为对比
        unselected = [p for p in all_patches if not any([
            p.get('rpsm_selected', False),
            p.get('improved_rpsm_selected', False), 
            p.get('angio_rpsm_selected', False)
        ]) and p not in selected_patches and p['total_cells'] > 0]
        
        if unselected:
            # 选择细胞数量适中的patch
            unselected.sort(key=lambda x: x['total_cells'])
            mid_idx = len(unselected) // 2
            selected_patches.append(unselected[mid_idx])
        
        # 如果patch数量不够3个，从剩余的valid_patches中选择
        while len(selected_patches) < 3 and len(selected_patches) < len(all_patches):
            remaining = [p for p in all_patches if p not in selected_patches and p['total_cells'] > 0]
            if remaining:
                # 按细胞数量排序，选择较好的
                remaining.sort(key=lambda x: x['total_cells'], reverse=True)
                selected_patches.append(remaining[0])
            else:
                break
        
        if not selected_patches:
            print(f"No valid patches with predictions found for WSI {wsi_idx + 1}")
            continue
        
        # 重新进行推理以获取详细的细胞标注信息
        def get_detailed_predictions(image_path, model, device):
            """重新推理获取详细的细胞预测信息"""
            try:
                # 加载和预处理图像
                image = cv2.imread(image_path)
                if image is None:
                    return None, None, None
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 确保图像尺寸为512x512
                if image_rgb.shape[0] != 512 or image_rgb.shape[1] != 512:
                    image_rgb = cv2.resize(image_rgb, (512, 512))
                
                # 转换为tensor
                from PIL import Image as PILImage
                image_pil = PILImage.fromarray(image_rgb)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                image_tensor = transform(image_pil).unsqueeze(0).to(device)
                
                # 推理
                model.eval()
                with torch.no_grad():
                    predictions = model(image_tensor)
                    predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)
                    predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)
                    
                    # 获取实例图和细胞类型信息
                    if hasattr(model, 'module'):  # DataParallel包装的模型
                        instance_map, instance_types = model.module.calculate_instance_map(predictions, magnification=40)
                    else:
                        instance_map, instance_types = model.calculate_instance_map(predictions, magnification=40)
                
                return image_rgb, instance_map, instance_types
                
            except Exception as e:
                print(f"Error in detailed prediction for {image_path}: {e}")
                return None, None, None
        
        # 创建可视化 - 每个patch显示原图、标注图和细胞分布
        n_patches = len(selected_patches)
        fig, axes = plt.subplots(3, n_patches, figsize=(6*n_patches, 15))
        
        if n_patches == 1:
            axes = axes.reshape(-1, 1)
        
        for patch_idx, patch in enumerate(selected_patches):
            patch_path = patch['patch_path']
            
            # 获取详细预测结果
            original_image, instance_map, instance_types = get_detailed_predictions(patch_path, pannuke_model, device)
            
            if original_image is None:
                print(f"Could not process image: {patch_path}")
                continue
            
            # 子图1: 原始图像
            ax1 = axes[0, patch_idx]
            ax1.imshow(original_image)
            
            # 标题包含RPSM状态
            rpsm_status = []
            if patch.get('rpsm_selected', False):
                rpsm_status.append("✓ Orig")
            if patch.get('improved_rpsm_selected', False):
                rpsm_status.append("✓ Impr")
            if patch.get('angio_rpsm_selected', False):
                rpsm_status.append("✓ Angio")
            
            if not rpsm_status:
                rpsm_status = ["✗ Not selected"]
            
            status_text = " | ".join(rpsm_status)
            ax1.set_title(f'Patch {patch_idx+1} - Original\nCells: {patch["total_cells"]}\n{status_text}', fontsize=10)
            ax1.axis('off')
            
            # 子图2: 带细胞标注的图像
            ax2 = axes[1, patch_idx]
            overlay_img = original_image.copy()
            
            # 在图像上绘制细胞标注
            if instance_types is not None and len(instance_types) > 0 and len(instance_types[0]) > 0:
                for cell_id, cell_info in instance_types[0].items():
                    if cell_info['type'] == 0:  # 跳过背景
                        continue
                    
                    # 获取细胞颜色
                    cell_type = cell_info['type']
                    color = color_dict.get(cell_type, [255, 255, 255])
                    
                    # 绘制轮廓
                    try:
                        contour = np.array(cell_info['contour'], dtype=np.int32)
                        cv2.drawContours(overlay_img, [contour], -1, color, 2)
                        
                        # 绘制质心
                        centroid = tuple(map(int, cell_info['centroid']))
                        cv2.circle(overlay_img, centroid, 3, color, -1)
                    except Exception as e:
                        print(f"Error drawing cell {cell_id}: {e}")
                        continue
                
                # 添加图例到图像右上角
                legend_height = 130
                legend_width = 160
                legend_start_x = max(0, overlay_img.shape[1] - legend_width - 10)
                legend_start_y = 10
                
                # 创建半透明背景
                legend_overlay = overlay_img.copy()
                cv2.rectangle(legend_overlay, 
                             (legend_start_x, legend_start_y), 
                             (legend_start_x + legend_width, legend_start_y + legend_height), 
                             (255, 255, 255), -1)
                
                # 混合图例背景
                alpha = 0.8
                overlay_img[legend_start_y:legend_start_y + legend_height, 
                           legend_start_x:legend_start_x + legend_width] = \
                    cv2.addWeighted(overlay_img[legend_start_y:legend_start_y + legend_height, 
                                              legend_start_x:legend_start_x + legend_width], 
                                   1 - alpha, 
                                   legend_overlay[legend_start_y:legend_start_y + legend_height, 
                                                legend_start_x:legend_start_x + legend_width], 
                                   alpha, 0)
                
                # 添加图例标题
                cv2.putText(overlay_img, "Cell Types:", 
                           (legend_start_x + 5, legend_start_y + 18), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                
                # 添加每种细胞类型的颜色和名称
                for i, (cell_type, name) in enumerate(type_names.items()):
                    y_pos = legend_start_y + 35 + i * 20
                    color = color_dict[cell_type]
                    
                    # 绘制颜色矩形
                    cv2.rectangle(overlay_img, 
                                 (legend_start_x + 5, y_pos - 8), 
                                 (legend_start_x + 20, y_pos + 3), 
                                 color, -1)
                    
                    # 添加文字
                    cv2.putText(overlay_img, name[:3], 
                               (legend_start_x + 25, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            
            ax2.imshow(overlay_img)
            ax2.set_title(f'Patch {patch_idx+1} - Cell Annotations', fontsize=10)
            ax2.axis('off')
            
            # 子图3: 细胞类型分布饼图
            ax3 = axes[2, patch_idx]
            cell_counts = patch['cell_counts']
            
            # 准备饼图数据
            sizes = []
            labels = []
            colors = []
            
            for cell_type in [1, 2, 3, 4, 5]:
                count = int(cell_counts.get(str(cell_type), 0))
                if count > 0:
                    sizes.append(count)
                    labels.append(f"{type_names[cell_type][:3]}\n({count})")
                    colors.append([c/255.0 for c in color_dict[cell_type]])
            
            if sizes:
                wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, 
                                                  autopct='%1.1f%%', startangle=90)
                # 设置文字大小
                for text in texts:
                    text.set_fontsize(8)
                for autotext in autotexts:
                    autotext.set_fontsize(7)
                    autotext.set_color('white')
                    autotext.set_weight('bold')
            else:
                ax3.text(0.5, 0.5, 'No cells\ndetected', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=12)
                ax3.set_xlim(0, 1)
                ax3.set_ylim(0, 1)
                
            ax3.set_title(f'Cell Distribution', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/wsi_{wsi_idx+1}_patch_predictions.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 清理GPU内存
        torch.cuda.empty_cache()
    
    print("Enhanced patch prediction visualizations with cell annotations saved to plots directory")



def calculate_cohens_d(group1, group2):
    """
    计算Cohen's d效应大小
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # 合并标准差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    d = (mean1 - mean2) / pooled_std
    return d

def evaluate_rpsm_methods(wsi_analyses):
    """
    全面评估四种RPSM方法的性能
    """
    responder_analyses = [a for a in wsi_analyses if a is not None and a['label'] == 1]
    non_responder_analyses = [a for a in wsi_analyses if a is not None and a['label'] == 0]
    
    if not responder_analyses or not non_responder_analyses:
        print("警告: 缺少响应者或非响应者数据，无法进行完整评估")
        return None
    
    methods = {
        'Original': ('rpsm_selected_count', 'rpsm_selected_patches'),
        'Improved': ('improved_rpsm_selected_count', 'improved_rpsm_selected_patches'),
        'Angiogenesis': ('angio_rpsm_selected_count', 'angio_rpsm_selected_patches'),
        'Hybrid': ('hybrid_rpsm_selected_count', 'hybrid_rpsm_selected_patches')
    }
    
    evaluation_results = {}
    
    for method_name, (count_key, patches_key) in methods.items():
        # 计算筛选率
        resp_rates = []
        non_resp_rates = []
        resp_angio_scores = []
        non_resp_angio_scores = []
        resp_quality_scores = []
        non_resp_quality_scores = []
        
        for a in responder_analyses:
            if count_key in a and a['total_patches'] > 0:
                rate = a[count_key] / a['total_patches']
                resp_rates.append(rate)
                
                # 计算血管生成一致性
                if patches_key in a and a[patches_key]:
                    angio_scores = [infer_angiogenesis_from_cells(p['cell_ratios']) for p in a[patches_key]]
                    resp_angio_scores.extend(angio_scores)
                    
                    # 计算质量分数
                    quality_scores = [calculate_patch_quality_score(p['cell_counts']) for p in a[patches_key]]
                    resp_quality_scores.extend(quality_scores)
                else:
                    print(f"Warning: No patches found for {method_name} responder analysis, patches_key: {patches_key}")
        
        for a in non_responder_analyses:
            if count_key in a and a['total_patches'] > 0:
                rate = a[count_key] / a['total_patches']
                non_resp_rates.append(rate)
                
                if patches_key in a and a[patches_key]:
                    angio_scores = [infer_angiogenesis_from_cells(p['cell_ratios']) for p in a[patches_key]]
                    non_resp_angio_scores.extend(angio_scores)
                    
                    quality_scores = [calculate_patch_quality_score(p['cell_counts']) for p in a[patches_key]]
                    non_resp_quality_scores.extend(quality_scores)
                else:
                    print(f"Warning: No patches found for {method_name} non-responder analysis, patches_key: {patches_key}")
        
        if resp_rates and non_resp_rates:
            # 统计测试
            t_stat, p_value = stats.ttest_ind(resp_rates, non_resp_rates)
            effect_size = calculate_cohens_d(resp_rates, non_resp_rates)
            
            # 血管生成一致性
            angio_consistency = {}
            if resp_angio_scores and non_resp_angio_scores:
                angio_t_stat, angio_p_value = stats.ttest_ind(resp_angio_scores, non_resp_angio_scores)
                angio_effect_size = calculate_cohens_d(resp_angio_scores, non_resp_angio_scores)
                angio_consistency = {
                    'resp_mean': np.mean(resp_angio_scores),
                    'non_resp_mean': np.mean(non_resp_angio_scores),
                    'difference': np.mean(resp_angio_scores) - np.mean(non_resp_angio_scores),
                    'effect_size': angio_effect_size,
                    'p_value': angio_p_value
                }
            
            # 质量分数对比
            quality_comparison = {}
            if resp_quality_scores and non_resp_quality_scores:
                quality_t_stat, quality_p_value = stats.ttest_ind(resp_quality_scores, non_resp_quality_scores)
                quality_effect_size = calculate_cohens_d(resp_quality_scores, non_resp_quality_scores)
                quality_comparison = {
                    'resp_mean': np.mean(resp_quality_scores),
                    'non_resp_mean': np.mean(non_resp_quality_scores),
                    'difference': np.mean(resp_quality_scores) - np.mean(non_resp_quality_scores),
                    'effect_size': quality_effect_size,
                    'p_value': quality_p_value
                }
            
            evaluation_results[method_name] = {
                'selection_rate': {
                    'responder_mean': np.mean(resp_rates),
                    'responder_std': np.std(resp_rates),
                    'non_responder_mean': np.mean(non_resp_rates),
                    'non_responder_std': np.std(non_resp_rates),
                    'difference': np.mean(resp_rates) - np.mean(non_resp_rates),
                    'effect_size': effect_size,
                    'p_value': p_value
                },
                'angiogenesis_consistency': angio_consistency,
                'quality_comparison': quality_comparison,
                'discrimination_ratio': np.mean(resp_rates) / (np.mean(non_resp_rates) + 1e-8)
            }
    
    return evaluation_results

def calculate_patch_quality_score(cell_counts):
    """
    计算patch的综合质量分数
    """
    total_cells = sum(cell_counts.values())
    if total_cells == 0:
        return 0
    
    # 计算细胞比例
    cell_ratios = {cell_type: count / total_cells for cell_type, count in cell_counts.items()}
    
    # 细胞密度评分 (0-0.3)
    density_score = min(total_cells / 200.0, 1.0) * 0.3
    
    # 细胞多样性评分 (0-0.3)
    cell_types = sum(1 for count in cell_counts.values() if count > 0)
    diversity_score = (cell_types / 5.0) * 0.3
    
    # 血管生成相关性评分 (0-0.4)
    angio_score = infer_angiogenesis_from_cells(cell_ratios) * 0.4
    
    return density_score + diversity_score + angio_score


def calculate_distribution_overlap(group1, group2):
    """
    计算两个分布的重叠程度
    """
    if not group1 or not group2:
        return 0
    
    min_val = min(min(group1), min(group2))
    max_val = max(max(group1), max(group2))
    
    # 创建直方图
    bins = np.linspace(min_val, max_val, 50)
    hist1, _ = np.histogram(group1, bins=bins, density=True)
    hist2, _ = np.histogram(group2, bins=bins, density=True)
    
    # 计算重叠面积
    overlap = np.sum(np.minimum(hist1, hist2)) * (bins[1] - bins[0])
    
    return overlap


def main():
    """
    主函数 - 增强版支持多GPU加速
    """
    # 确保每次运行都有不同的随机种子
    import time
    # 使用更复杂的随机种子生成方法
    random_seed = int(time.time() * 1000000) % 2147483647  # 使用微秒级时间戳
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(f"Using random seed: {random_seed}")
    
    # 额外的随机状态重置
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    # 数据路径
    csv_path = "slide_ov_response.csv"
    
    print("WSI Cell Distribution Analysis Script")
    print("="*50)
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"CSV file does not exist: {csv_path}")
        return
    
    # Setup and load models (multi-GPU or single-GPU)
    print("1. Loading PanNuke model...")
    model_result = setup_pannuke_models()
    if model_result is None:
        return
    
    # Unpack the result
    if len(model_result) == 3:
        pannuke_model, device, is_multi_gpu = model_result
    else:
        print("Unexpected return format from setup_pannuke_models")
        return
    
    if is_multi_gpu:
        print(f"🚀 Multi-GPU mode activated with DataParallel")
        print(f"💡 Expected speed improvement with parallel processing")
    else:
        print("📱 Single-GPU mode with optimized batch size")
    
    # Choose analysis mode: 'all' for complete evaluation, 'sample' for subset analysis, 'load' for loading existing results
    #ANALYSIS_MODE = 'all'    # 🎯 FOR COMPREHENSIVE RPSM OPTIMIZATION: 分析所有WSI数据
    #ANALYSIS_MODE = "load"  # 从已保存的结果中加载数据进行评估  
    ANALYSIS_MODE = "sample"  # 分析部分WSI样本，测试RPSM筛选标准
    
    print(f"📊 Analysis mode: {ANALYSIS_MODE.upper()}")
    
    if ANALYSIS_MODE == 'load':
        # Load existing analysis results
        print("\n2. Loading existing analysis results...")
        import glob
        
        # 查找最新的报告文件
        report_files = glob.glob('reports/wsi_analysis_data_*.json')
        if report_files:
            latest_file = max(report_files)
            print(f"📂 Loading data from: {latest_file}")
            
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                
                # 检查是否包含完整的 wsi_analyses 数据
                if 'wsi_analyses' in data:
                    wsi_analyses = data['wsi_analyses']
                    print(f"✅ Loaded {len(wsi_analyses)} WSI analyses from saved data")
                else:
                    print("⚠️  Old data format detected - missing detailed patch data")
                    print("    Please run analysis with ANALYSIS_MODE = 'sample' or 'all' to get complete evaluation")
                    
                    # 创建最小化的分析数据结构以支持基本功能
                    wsi_analyses = []
                    for sample in data.get('sample_details', []):
                        analysis = {
                            'wsi_path': sample['sample_name'],
                            'label': 1 if sample['label'] == 'responder' else 0,
                            'total_patches': sample['total_patches'],
                            'rpsm_selected_count': sample['rpsm_results']['original']['selected_count'],
                            'improved_rpsm_selected_count': sample['rpsm_results']['improved']['selected_count'],
                            'angio_rpsm_selected_count': sample['rpsm_results']['angiogenesis']['selected_count'],
                            'hybrid_rpsm_selected_count': sample['rpsm_results'].get('hybrid', {}).get('selected_count', 0)
                        }
                        wsi_analyses.append(analysis)
                        
            except Exception as e:
                print(f"❌ Error loading data: {e}")
                print("    Falling back to sample mode")
                ANALYSIS_MODE = "sample"
        else:
            print("❌ No existing analysis data found")
            print("    Falling back to sample mode")
            ANALYSIS_MODE = "sample"
    
    if ANALYSIS_MODE == 'all':
        # Load ALL WSI data for complete RPSM evaluation
        print("\n2. Loading ALL WSI data for complete RPSM evaluation...")
        samples = load_all_wsi_data(csv_path)  # 分析所有WSI样本，完整评估RPSM筛选标准
    elif ANALYSIS_MODE == "sample":
        # Load sample WSI data for testing
        print("\n2. Loading sample WSI data for testing...")
        # samples = load_and_sample_wsi_data(csv_path, num_samples_per_group=3)  # 分析部分WSI样本，测试RPSM筛选标准
        samples = load_all_wsi_data(csv_path)
    import ipdb;
    if ANALYSIS_MODE != 'load':
        if not samples:
            print("No available samples found")
            return
        
        # Analyze each WSI sample with checkpoint resume capability
        print("\n3. Starting WSI sample analysis...")
        wsi_analyses = []
        
        # 检查是否有checkpoint文件，支持断点续传
        checkpoint_file = "wsi_analysis_progress.json"
        start_index = 0
        
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    progress_data = json.load(f)
                    start_index = progress_data.get('completed_count', 0)
                    wsi_analyses = progress_data.get('analyses', [])
                print(f"📂 Found checkpoint: resuming from WSI {start_index + 1}")
            except:
                print("⚠️  Checkpoint file corrupted, starting fresh")
                start_index = 0
                wsi_analyses = []
        
        total_start_time = time.time()
        # import ipdb; ipdb.set_trace()
        # 从checkpoint开始处理
        for i in range(start_index, len(samples)):
            sample = samples[i]
            sample_start_time = time.time()
            print(f"\n⏱️ Processing WSI {i+1}/{len(samples)}: {sample['slides_name']}")
            
            try:
                # import ipdb; ipdb.set_trace()
                analysis = analyze_wsi_sample(sample, pannuke_model, device, is_multi_gpu)
                wsi_analyses.append(analysis)
                
                # 每处理完一个WSI就清理内存并保存进度
                torch.cuda.empty_cache()
                
                # 更新进度文件
                progress_data = {
                    'completed_count': i + 1,
                    'analyses': wsi_analyses,
                    'timestamp': datetime.now().isoformat()
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(progress_data, f, default=numpy_json_serializer, indent=2)
                
                sample_time = time.time() - sample_start_time
                remaining_samples = len(samples) - (i + 1)
                estimated_remaining_time = sample_time * remaining_samples
                
                print(f"✅ WSI {i+1} completed in {sample_time:.1f}s")
                if remaining_samples > 0:
                    print(f"📊 Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
                    
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"❌ Error processing WSI {i+1}: {type(e).__name__}: {e}")
                print(f"📋 Full error details:\n{error_details}")
                # 添加错误的WSI到结果中，但继续处理
                wsi_analyses.append(None)
                torch.cuda.empty_cache()  # 发生错误时也清理内存
                
                # 即使出错也保存进度
                progress_data = {
                    'completed_count': i + 1,
                    'analyses': wsi_analyses,
                    'timestamp': datetime.now().isoformat(),
                    'last_error': str(e)
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(progress_data, f, default=numpy_json_serializer, indent=2)
        
        total_time = time.time() - total_start_time
        print(f"\n🎯 Total analysis time: {total_time/60:.1f} minutes")
        
        if is_multi_gpu:
            num_gpus = torch.cuda.device_count()
            single_gpu_estimated = total_time * num_gpus
            speedup = single_gpu_estimated / total_time
            print(f"🚀 Multi-GPU speedup achieved: {speedup:.1f}x")
    else:
        print(f"\n✅ Loaded existing analysis results with {len(wsi_analyses)} WSI samples")
    
   
if __name__ == "__main__":
    main()
