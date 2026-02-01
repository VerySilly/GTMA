import gradio as gr
import os, requests
import numpy as np
import torch
import cv2
from cell_segmentation.inference.cell_detection import CellSegmentationInference
from cell_segmentation.inference.inference_cellvit_experiment_pannuke import InferenceCellViTParser,InferenceCellViT
from cell_segmentation.inference.inference_cellvit_experiment_monuseg import InferenceCellViTMoNuSegParser,MoNuSegInference

import pandas as pd
from skimage.measure import regionprops
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial import Delaunay
from sklearn.preprocessing import StandardScaler
import copy
import tqdm
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
        for inst_id in tqdm.tqdm(range(1, self.n_instances+1)):
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

## local |  remote
RUN_MODE = "remote"
if RUN_MODE != "local":
    # Only download if files don't exist
    if not os.path.exists("checkpoints/model_best.pth"):
        os.system("wget https://hf-mirror.com/xiazhi/LKCell/resolve/main/model_best.pth")
        os.system("mkdir -p checkpoints && mv model_best.pth checkpoints/")
    
    ## examples
    if not os.path.exists("1.png"):
        os.system("wget https://hf-mirror.com/xiazhi/LKCell/resolve/main/1.png")
    if not os.path.exists("2.png"):
        os.system("wget https://hf-mirror.com/xiazhi/LKCell/resolve/main/2.png")
    if not os.path.exists("3.png"):
        os.system("wget https://hf-mirror.com/xiazhi/LKCell/resolve/main/3.png")
    if not os.path.exists("4.png"):
        os.system("wget https://hf-mirror.com/xiazhi/LKCell/resolve/main/4.png")

## step 1: set up model

device = "cuda:0"

## pannuke set
pannuke_parser = InferenceCellViTParser()
pannuke_configurations = pannuke_parser.parse_arguments()
pannuke_inf = InferenceCellViT(
        run_dir=pannuke_configurations["run_dir"],
        checkpoint_name=pannuke_configurations["checkpoint_name"],
        gpu=pannuke_configurations["gpu"],
        magnification=pannuke_configurations["magnification"],
    )

pannuke_checkpoint = torch.load(
    pannuke_inf.run_dir / "checkpoints" / pannuke_inf.checkpoint_name, map_location="cpu"
)
pannuke_model = pannuke_inf.get_model(model_type=pannuke_checkpoint["arch"])
pannuke_model.load_state_dict(pannuke_checkpoint["model_state_dict"])
# # put model in eval mode
pannuke_model.to(device)
pannuke_model.eval()


## monuseg set
monuseg_parser = InferenceCellViTMoNuSegParser()
monuseg_configurations = monuseg_parser.parse_arguments()
monuseg_inf = MoNuSegInference(
        model_path=monuseg_configurations["model"],
        dataset_path=monuseg_configurations["dataset"],
        outdir=monuseg_configurations["outdir"],
        gpu=monuseg_configurations["gpu"],
        patching=monuseg_configurations["patching"],
        magnification=monuseg_configurations["magnification"],
        overlap=monuseg_configurations["overlap"],
    )


def click_process(image_input , type_dataset):
    import torch
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    import io
    # Color mapping for different nuclei types
    color_dict = {
        0: [0, 0, 0],       # Background - black
        1: [255, 0, 0],     # Neoplastic - red  
        2: [0, 255, 0],     # Inflammatory - green
        3: [0, 0, 255],     # Connective - blue
        4: [255, 255, 0],   # Dead - yellow
        5: [255, 0, 255],   # Epithelial - magenta
    }
    
    if type_dataset == "pannuke":
        if image_input.shape[0] != 512 and image_input.shape[1] != 512:
            image_input = cv2.resize(image_input, (512,512))
        
        # Transform image for inference
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Convert to tensor and add batch dimension
        image_tensor = transform(image_input).unsqueeze(0).to(device)
        
        # Run inference with already loaded model
        with torch.no_grad():
            predictions = pannuke_model(image_tensor)
        
        # Apply softmax to get probabilities
        predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)
        predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)
        
        # Get cell predictions with tokens
        instance_map, instance_types = pannuke_model.calculate_instance_map(predictions, magnification=40)
        
        # Create visualizations
        binary_map = predictions["nuclei_binary_map"][0].cpu().numpy()
        nuclei_type_map = predictions["nuclei_type_map"][0].cpu().numpy()
        hv_map = predictions["hv_map"][0].cpu().numpy()
        instance_map = instance_map[0].cpu().numpy()
        
        # Create overlay visualization on original image
        overlay_img = image_input.copy()
        
        # Draw cells on overlay
        if len(instance_types) > 0:
            for cell_id, cell_info in instance_types[0].items():
                if cell_info['type'] == 0:  # Skip background
                    continue
                    
                # Get cell color
                cell_type = cell_info['type']
                color = color_dict.get(cell_type, [255, 255, 255])
                
                # Draw contour
                contour = np.array(cell_info['contour'], dtype=np.int32)
                cv2.drawContours(overlay_img, [contour], -1, color, 2)
                
                # Draw centroid
                centroid = tuple(map(int, cell_info['centroid']))
                cv2.circle(overlay_img, centroid, 3, color, -1)
        
        # Add legend directly on the overlay image at top-right corner
        legend_height = 130
        legend_width = 160
        legend_start_x = overlay_img.shape[1] - legend_width - 10
        legend_start_y = 10
        
        # Create semi-transparent background for legend
        legend_overlay = overlay_img.copy()
        cv2.rectangle(legend_overlay, 
                     (legend_start_x, legend_start_y), 
                     (legend_start_x + legend_width, legend_start_y + legend_height), 
                     (255, 255, 255), -1)
        
        # Blend the legend background with the original image
        alpha = 0.8
        overlay_img[legend_start_y:legend_start_y + legend_height, 
                   legend_start_x:legend_start_x + legend_width] = \
            cv2.addWeighted(overlay_img[legend_start_y:legend_start_y + legend_height, 
                                      legend_start_x:legend_start_x + legend_width], 
                           1 - alpha, 
                           legend_overlay[legend_start_y:legend_start_y + legend_height, 
                                        legend_start_x:legend_start_x + legend_width], 
                           alpha, 0)
        
        # Add legend text and colors
        type_names = {1: "Neoplastic", 2: "Inflammatory", 3: "Connective", 4: "Dead", 5: "Epithelial"}
        
        # Add title
        cv2.putText(overlay_img, "Cell Types:", 
                   (legend_start_x + 5, legend_start_y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        
        # Add each cell type with color
        for i, (cell_type, name) in enumerate(type_names.items()):
            y_pos = legend_start_y + 35 + i * 20
            color = color_dict[cell_type]
            
            # Draw color rectangle
            cv2.rectangle(overlay_img, 
                         (legend_start_x + 5, y_pos - 8), 
                         (legend_start_x + 20, y_pos + 3), 
                         color, -1)
            
            # Add text
            cv2.putText(overlay_img, name, 
                       (legend_start_x + 25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        
        cv2.imwrite("raw_pred.png", cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        
        # Create comprehensive prediction visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0,0].imshow(image_input)
        axes[0,0].set_title('Image')
        axes[0,0].axis('off')
        
        # Binary nuclei prediction
        axes[0,1].imshow(binary_map[1], cmap='gray')
        axes[0,1].set_title('Binary-Cells')
        axes[0,1].axis('off')
        
        # HV Map - Horizontal
        axes[0,2].imshow(hv_map[0], cmap='viridis')
        axes[0,2].set_title('HV-Map-0')
        axes[0,2].axis('off')
        
        # HV Map - Vertical  
        axes[1,0].imshow(hv_map[1], cmap='viridis')
        axes[1,0].set_title('HV-Map-1')
        axes[1,0].axis('off')
        
        # Instance map
        axes[1,1].imshow(instance_map, cmap='tab20')
        axes[1,1].set_title('Instances')
        axes[1,1].axis('off')
        
        # Nuclei type prediction
        nuclei_pred = np.argmax(nuclei_type_map, axis=0)
        axes[1,2].imshow(nuclei_pred, cmap='tab10')
        axes[1,2].set_title('Nuclei-Pred')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.savefig("pred_img.png", dpi=150, bbox_inches='tight')
        plt.close()
        
    else:
        if image_input.shape[0] != 512 and image_input.shape[1] != 512:
            image_input = cv2.resize(image_input, (512,512))
        
        # Transform image for inference
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        image_tensor = transform(image_input).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = monuseg_inf.model(image_tensor)
        
        predictions = monuseg_inf.get_cell_predictions(predictions)
        
        # Create visualizations
        binary_map = predictions["nuclei_binary_map"][0].cpu().numpy()
        instance_map = predictions["instance_map"][0].cpu().numpy()
        
        # Create overlay visualization
        overlay_img = image_input.copy()
        
        # Draw instance contours
        for instance_id in np.unique(instance_map):
            if instance_id == 0:
                continue
            mask = (instance_map == instance_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_img, contours, -1, (0, 255, 0), 2)
        
        cv2.imwrite("raw_pred.png", cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        
        # Create comprehensive visualization

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_input)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(binary_map[1], cmap='gray')
        axes[1].set_title('Binary Prediction')
        axes[1].axis('off')
        
        axes[2].imshow(instance_map, cmap='tab20')
        axes[2].set_title('Instance Map')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig("pred_img.png", dpi=150, bbox_inches='tight')
        plt.close()
    # import ipdb
    # ipdb.set_trace()
    slide_obj = SlideNucStatObject(instance_map=instance_map,inst_type = nuclei_pred, image=image_input)
    df_features = slide_obj.compute_features()
    image_output = cv2.imread("raw_pred.png")
    image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2RGB)     
    image_output2 = cv2.imread("pred_img.png")
    image_output2 = cv2.cvtColor(image_output2, cv2.COLOR_BGR2RGB)   
    
    return image_output, image_output2, df_features

def download_csv(df):
    import io
    csv_bytes = df.to_csv(index=False).encode()
    csv_file = io.BytesIO(csv_bytes)
    csv_file.name = "nuclei_features.csv"
    return csv_file
# def get_csv(df):
#     import tempfile
#     tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
#     df.to_csv(tmp_file.name, index=False)
#     tmp_file.close()s
#     tmp_file.name = 'nuclei_features.csv'
#     return tmp_file.name
def save_csv_to_local(df):
    # 确保 df 是 DataFrame
    df = pd.DataFrame(df)
    
    # 写入当前目录
    file_path = "nuclei_features.csv"
    df.to_csv(file_path, index=False, encoding="utf-8")
    
    return f"Saved CSV to {file_path}"



demo = gr.Blocks(title="CPG-MIL")
with demo:
    gr.Markdown(value="""
                    **Gradio demo for LKCell: Efficient Cell Nuclei Instance Segmentation with Large Convolution Kernels**. Check our [Github Repo](https://github.com/hustvl/LKCell) 😛.
                    """)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                Image_input = gr.Image(type="numpy", label="Input", interactive=True,height=480)
            with gr.Row():
                Type_dataset = gr.Radio(choices=["pannuke", "monuseg"], label=" input image's dataset type",value="pannuke")
            with gr.Row():
                dataframe_output = gr.Dataframe(label="Nuclei Features", interactive=False)
                                
        with gr.Column():
            image_output = gr.Image(type="numpy", label="image prediction",height=480,width=480)
            image_output2 = gr.Image(type="numpy", label="all predictions",height=480)
            
    with gr.Row():
        Button_run = gr.Button("🚀 Submit (发送) ")
        clear_button = gr.ClearButton(components=[Image_input,Type_dataset,image_output,image_output2],value="🧹 Clear (清除)")
        
    with gr.Row():

        download_button = gr.Button("📥 Download CSV (下载CSV)")

    Button_run.click(fn=click_process, inputs=[Image_input,  Type_dataset ], outputs=[image_output,image_output2,dataframe_output])
    # download_button.click(fn=download_csv, inputs=[dataframe_output], outputs=[gr.File(label="Download CSV")])

    download_button.click(fn=save_csv_to_local, inputs=[dataframe_output], outputs=[download_button])

    
    ## guiline
    gr.Markdown(value="""    
                    🔔**Guideline**
                    1. Upload your image or select one from the examples.
                    2. Set up the arguments: "Type_dataset" to enjoy two dataset type's inference
                    3. Due to the limit of CPU , we resize the input image whose size is larger than (512,512) to (512,512)
                    4. Run the Submit button to get the output.
                    """)
    # if RUN_MODE != "local":
    gr.Examples(examples=[
                ['1.png', "pannuke"],
                ['2.png', "pannuke"],
                ['3.png', "monuseg"],
                ['4.png', "monuseg"],
                ], 
                inputs=[Image_input, Type_dataset], outputs=[image_output,image_output2], label="Examples")
    gr.HTML(value="""
                <p style="text-align:center; color:orange"> <a href='https://github.com/hustvl/LKCell' target='_blank'>Github Repo</a></p>
                    """)
    gr.Markdown(value="""    
                    Template is adapted from [Here](https://huggingface.co/spaces/menghanxia/disco)
                    """)

if RUN_MODE == "local":
    demo.launch(server_name='127.0.0.1',server_port=8003)
else:
    demo.launch()