<p align="center" style="text-align: center;">
  <img src="https://github.com/PaMoSplat-old/anonymous_code/blob/main/logo.svg" alt="PaMoSplat Logo" style="vertical-align: bottom; width:300px;"/>
</p>

<p align="center">
  <h1 align="center"><strong>
    Part-Aware Motion-Guided Gaussian Splatting for Dynamic Scene Reconstruction
  </strong>
</p>


<p align="center">
  <a href="https://pamosplat.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-👔-green?">
  </a> 
</p>


 ## 🏠  Abstract
Dynamic scene reconstruction is a critical yet challenging task in both computer vision and robotics. 
Despite recent advancements in 3D Gaussian Splatting (3DGS) based approaches for modeling dynamics, achieving high-quality rendering and precise tracking in large complex motion scenes remains formidable.
To address these challenges, we propose PaMoSplat, a novel Gaussian splatting framework incorporating part awareness and motion priors. 
Two key insights of PaMoSplat are: 1) Parts serve as primitives for scene deformation, and 2) Motion cues from optical flow can effectively guide movements.
In PaMoSplat, for the initial timestamp, graph clustering technique facilitates the lifting of multi-view segmentation masks into 3D to create Gaussian parts. For subsequent timestamps, a differential evolutionary algorithm is employed to infer prior motion of these Gaussian parts based on the optical flow across views, serving as initial state for further optimization. Additionally, PaMoSplat introduces internal learnable rigidity for the parts and flow-supervised rendering loss.
Experiments on various scenes demonstrate that PaMoSplat achieves exceptional rendering quality and tracking accuracy. Furthermore, it enables part-level downstream applications, including 4D video editing.
 

## 🛠  Install


### Clone this repo

```bash
git clone git@github.com:BIT-DYN/pamosplat.git
cd pamosplat
```


### Install the required libraries
Use conda to install the required environment. It is recommended to follow the instructions below to set up the environment.


```bash
conda env create -f environment.yml
```

### Install SAM Model (for segement in 2D)
Follow the [instructions](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#installation) to install the SAM model and download the pretrained weights.


###  Install TAP Model (for gen caption for 2D mask)
Follow the [instructions](https://github.com/baaivision/tokenize-anything?tab=readme-ov-file#installation) to install the TAP model and download the pretrained weights [here](https://github.com/baaivision/tokenize-anything?tab=readme-ov-file#models).


###  Install SBERT Model (for query part in caption feature)
```bash
pip install -U sentence-transformers
```
Download pretrained weights
```bash
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```


###  Install RAFT Model (for generate optical flow)
Download pretrained weights from [instructions](https://github.com/princeton-vl/RAFT?tab=readme-ov-file#demos)

### Install rendering code
```bash
git clone git@github.com:JonathonLuiten/diff-gaussian-rasterization-w-depth.git
cd diff-gaussian-rasterization-w-depth
python setup.py install
pip install .
```


## 📊 Prepare dataset
OpenGraph has completed validation on [PanopticSport](https://github.com/JonathonLuiten/Dynamic3DGaussians), [ParticleNeRF](https://github.com/jc211/ParticleNeRF) and self-captured dataset. 

We organize all of our datasets in the format of PanopticSport, so we recommend downloading it for quick use.

* [PanopticSport](https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip) - All six sequences in PanopticSport.
* [ParticleNeRF](https://zenodo.org/records/7784157) - Some sequences in ParticleNeRF, we used pendulums and robot_task in paper.
* Self-captured dataset - We will be releasing it soon.

## 🏃 Run

### 2D Mask Generation
Run the following command to to genearte 2D mask in the initial timestamp. 

The result will be saved in ```data/{sequence}/mask``` folder, where ```sequence``` is the sequence you used in this code and defaults to ```softball```.

```bash
cd gen_mask
python sam_mask.py
```

### 2D Flow Generation
Run the following command to to genearte 2D optical flow in the subsequent timestamps. 

The result will be saved in ```data/{sequence}/for_flow``` and ```data/{sequence}/rev_flow``` folders, where ```for_flow``` represents forward optical flow and ```rev_flow``` represents reverse optical flow.

```bash
cd ..
cd gen_flow --model=/code1/dyn/codes/RAFT/models/raft-things.pth
python raft_flow.py
```


### Train
Run the following command to begin training in default sequence ```softball```.
```bash
cd ..
python train.py
```
The result will be saved in ```output/PaMoSplat/{sequence}``` floder.


### Visulization
Interactions can be made using our visualization files.

At the beginning of the code, we provide several visualization modes and you can select them manually.
```bash
python visualize.py
```