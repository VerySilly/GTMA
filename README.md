# Interpretable graph learning of spatial TME patterns in digital pathology for gastric cancer prognosis

A graph-based deep learning framework for predicting survival from whole slide images.

## Dependencies

* To install the dependencies for GTMA project, see the "GATM_requirements.yaml"

  To install the dependencies for LKcell project, see the "LKcell_requirements.yaml"

```
conda env create -f GATM_requirements.yml
```

## Step 1: Processing whole slide image (WSI) into superpatch-graph

#### What is the superpatch-graph?

* Superpatch-graph is the compressed representation of whole slide image into graph structure in memory efficient manner.

  * Users can simply run the above script with pre-defined sample data"GTMA/data"

  * Or, users can use your own whole slide image by setting the "--graphdir"

  * Output files

    * Compressed network as ".pt"

    * Node position information in "_node_location_list.csv"

    * Superpatch aggregated dictionary in "_artifact_sophis_final.csv"

```
cd Superpatch_network_construction
python supernode_generation.py \
	--graphdir <path_save_graph>\
   --imagedir <svs_file>
```

## Step 2: Training GTMA using superpatch-graph

* Users can predict the prognosis of entire host with tumor environment-associated context analysis using GTMA

* Run the ./main.py with appropriate hyperparameters

  * Users can simply run the above script with pre-defined parameters and datasets

  * Or, users can use own dataset preprocessed by "supernode_generation" script

```
python main.py

```

## Step 3: Visualization of IG (Integrated gradients) value on WSI

* Users can visualize the IG value which is highly correlated with risk value of each region in WSI

  * Users must define the trained_parameters as "--load_state_dict"

```
python IG_attention_feature_cal_main.py\
	--load_state_dict <path_to_best_pt>
```

## Step 4: Nuclei segmentation and cell features

* Users can identify cell composition using the LKcell model and extracted feature&#x20;

```
conda env create -fLKcell_environment.yml
conda activate LKcell
python
```

* Users can visualize cell features using the web

```
python app.py
```

## Users can identify cell composition using the LKcell model and extracted featureAcknowledgments

* http://github.com/mahmoodlab/Patch-GCN

* http://github.com/lukemelas/EfficientNet-PyTorch

* http://github.com/pyg-team/pytorch_geometric

* https://github.com/hustvl/LKCell

