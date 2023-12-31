# APTM


**APTM** is a new joint **A**ttribute **P**rompt Learning and **T**ext **M**atching Learning framework, considering the shared knowledge between attribute and text. As the name implies, APTM contains an attribute prompt learning stream and a text matching learning stream.

We also present a large Multi-Attribute and Language Search dataset for text-based person retrieval, called **MALS**, and explore the feasibility of performing pre-training on both attribute recognition and image-text matching tasks in one stone. In particular, MALS contains 1, 510, 330 image-text pairs, which is about 37.5× larger than prevailing CUHK-PEDES, and all images are annotated with 27 attributes. 

Extensive experiments validate the effectiveness of the pre-training on MALS, achieving the state-of-the-art retrieval performance via APTM on three challenging real-world benchmarks. In particular, APTM achieves a consistent improvement of +6.60%, +7.39%, and +15.90% Recall@1 accuracy on CUHK-PEDES, ICFG-PEDES, and RSTPReid datasets by a clear margin, respectively. More details can be found at our paper: [Towards Unified Text-based Person Retrieval: A Large-scale Multi-Attribute and Language Search Benchmark](https://arxiv.org/abs/2306.02898)
<div align="center"><img src="assets/framework.jpg" width="600"></div>

## News
* The **APTM** is released. Welcome to communicate！

## MALS
MALS leverages generative models to generate a large-scale dataset including 1.5𝑀 image-text pairs. Each image-text pair in MALS is annotated with one corresponding description and several appropriate attribute labels, indicating that MALS is not only effective for text-image matching and attribute prompt learning, but also explores the feasibility of pre-training for both attribute recognition and image-text matching in one stone.The dataset is released at [Baidu Yun](link1). 

**Note that MALS can only be used for research, any commercial usage is forbidden.**

This is the comparison between MALS and other text based person retrieval datasets. 
<div align="center"><img src="assets/chart1.jpg" width="900"></div>
These are examples of our MALS dataset and CUHK-PEDES.
<div align="center"><img src="assets/examples.jpg" width="900"></div>
Annotation format:

```
[{
"image": "gene_crop/c_g_a_0/0.jpg",
"caption": "a young boy wearing a black hoodie leaning against a wall with his hands on his hips and his hands on his hips wearing jeans and a baseball cap",
 "image_id": "c_g_a_0_0"},
...
{"image": "gene_crop/c_g_a_0/20217.jpg",
"caption": "a woman in a white top and black pants posing for a picture in front of a brick wall with a pink carpet in front of her", "image_id": "c_g_a_0_20217"}
]
```

## Models and Weights

The checkpoints have been released at [Baidu Yun](link1) and [Google Drive](link2)


## Usage

### Install Requirements

we use 4 A100 80G GPU for training and evaluation.

Create conda environment.

```
conda create --name APTM --file requirements.txt
conda activate APTM
```

### Datasets Prepare

Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) , the pa100k dataset from [here](https://github.com/xh-liu/HydraPlus-Net), the RSTPReid dataset from [here](https://github.com/NjtechCVLab/RSTPReid-Dataset), and ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN).
[swin_base_patch4_window7_224_22k.pth](link) 
[gene_attrs](link)

Organize `data` folder as follows:

```
|-- data/
|		|-- bert-base-uncased
|		|-- finetune
|       |-- gene_attrs
|            |-- g_4x_attrs.json
|            |-- g_c_g_a_0_attrs.json
|            |-- ...
|            |-- g_c_g_a_7_attrs.json
|		|-- swin_base_patch4_window7_224_22k.pth
```

And organize those datasets in `images` folder as follows:

```
|-- images/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|
|   |-- <pa100k>/
|       |-- release_data
|       |-- ...
|   |-- <RSTPReid>/
|       |-- ...
```

### Pretraining  Inference
We pretrain our APTM using MALS as follows：

```
python3 run.py --task "itr_gene" --dist "f4" --output_dir "/home/x_out/3w_attr_lb0.4_0.8" --checkpoint "16m_base_model_state_step_199999.th"
```

### Fine-tuning Inference
We fine-tune our APTM using existing person-reid datasets. Performance can be improved through replacing the backbone with our pre-trained model. Taking CUHK-PEDES as example:

```
python3 run.py --task "itr_cuhk" --dist "f4" --output_dir "/home/x_out/151w_attr_lb0.4_0.8/cuhk_eda_b120" --checkpoint "/home/x_out/151w_attr_lb0.4_0.8/checkpoint_31.pth"
```

### Evaluation

```
python3 run.py --task "itr_cuhk" --evaluate --dist "f4" --output_dir "output/baseline/itc_itm_mlm/150w/t2i" --checkpoint "output/baseline/itc_itm_mlm/150w/checkpoint_best.pth"
```

## Reference
If you use APTM in your research, please cite it by the following BibTeX entry:

```
@article{yang2023towards,
  title={Towards Unified Text-based Person Retrieval: A Large-scale Multi-Attribute and Language Search Benchmark},
  author={Yang, Shuyu and Zhou, Yinan and Wang, Yaxiong and Wu, Yujiao and Zhu, Li and Zheng, Zhedong},
  journal={arXiv preprint arXiv:2306.02898},
  year={2023}
}

```
