# SITS-Former: A Pre-Trained Spatio-Spectral-Temporal Representation Model for Sentinel-2 Time Series Classification
PyTorch implementation of the pre-trained model presented in "***SITS-Former: A Pre-Trained Spatio-Spectral-Temporal Representation Model for Sentinel-2 Time Series Classification***" published in International Journal of Applied Earth Observation and Geoinformation.

## Citation and Contact
If you use our code or dataset, please cite to the following paper:
```
@article{yuan2022sits,
  title={SITS-Former: A pre-trained spatio-spectral-temporal representation model for Sentinel-2 time series classification},
  author={Yuan, Yuan and Lin, Lei and Liu, Qingshan and Hang, Renlong and Zhou, Zeng-Guang},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={106},
  pages={102651},
  year={2022},
  doi = {10.1016/j.jag.2021.102651},
  publisher={Elsevier}
}
```

## Abstract
Sentinel-2 images provide a rich source of information for a variety of land cover, vegetation, and environmental monitoring applications due to their high spectral, spatial, and temporal resolutions. Recently, deep learning-based classification of Sentinel-2 time series becomes a popular solution to vegetation classification and land cover mapping, but it often demands a large number of manually annotated labels. Improving classification performance with limited labeled data is still a challenge in many real-world remote sensing applications. To address label scarcity, we present SITS-Former (SITS stands for Satellite Image Time Series and Former stands for Transformer), a pre-trained representation model for Sentinel-2 time series classification. SITS-Former adopts a Transformer encoder as the backbone and takes time series of image patches as input to learn spatio-spectral-temporal features. According to the principles of self-supervised learning, we pre-train SITS-Former on massive unlabeled Sentinel-2 time series via a missing-data imputation proxy task. Given an incomplete time series with some patches being masked randomly, the network is asked to regress the central pixels of these masked patches based on the residual ones. By doing so, the network can capture high-level spatial and temporal dependencies from the data to learn discriminative features. After pre-training, the network can adapt the learned features to a target classification task through fine-tuning. As far as we know, this is the first study that exploits self-supervised learning for patch-based representation learning and classification of SITS. We quantitatively evaluate the quality of the learned features by transferring them on two crop classification tasks, showing that SITS-Former outperforms state-of-the-art approaches and yields a significant improvement (2.64%~3.30% in overall accuracy) over the purely supervised model. The proposed model provides an effective tool for SITS-related applications as it greatly reduces the burden of manual labeling.

## Requirements
+ tqdm = 4.62.2
+ numpy = 1.19.2
+ python >= 3.8
+ pytorch = 1.7.1
+ tensorboard = 2.4.0
+ pandas = 1.0.3
+ sklearn = 1.0.1

## Datasets
* The complete **unlabeled dataset** for model pre-training can be downloaded [here](https://zenodo.org/record/5803021#.YdFIB2hBzcv). If you use this dataset in your research, please cite our paper.
* A large open-access benchmark **S2-2017-T31TFM** intrdouced by V. Sainte Fare Garnot is used for method evaluation, which can be downloaded [here](https://zenodo.org/record/5815523). Please run the provided code `PrepareData.ipynb` to process the data into acceptable formats. Note that the copyright of this dataset belongs to its producer. If you use this dataset in your research, please cite its DOI and the related paper: **Garnot V S F, Landrieu L, Giordano S, et al. Satellite image time series classification with pixel-set encoders and temporal self-attention[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 12325-12334.**

## Pre-Training SITS-Former

The file `pretraining.py` is the main code for pre-training a SITS-Former model. All the unlabeled time series used in the pre-training stage are stored in the folder `Unlabeled-Data-PixelPatch`. The scrpit can automatically split the data into training and validation sets according to the given *valid_rate*.

You can run the following Linux command to pretrain SITS-Former from scratch using our recommend hyperparameters.
```
python pretraining.py \   
    --dataset_path '../Unlabeled-Data-PixelPatch' \
    --pretrain_path '../checkpoints/pretrain' \
    --num_workers 8 \
    --valid_rate 0.03 \
    --max_length 50 \
    --patch_size 5 \
    --num_features 10 \
    --mask_rate 0.3 \
    --epochs 200 \
    --batch_size 512 \
    --hidden_size 256 \
    --layers 3 \
    --attn_heads 8 \
    --learning_rate 1e-3 \
    --warmup_epochs 10 \
    --decay_gamma 0.99 \
    --dropout 0.1 \
    --gradient_clipping 5.0
```

We also provide the pre-trained model parameters of SITS-Former, which are available in the `checkpoints\pretrain\` folder.

## Fine-Tuning SITS-Former

The file `finetuning.py` is the main code for fine-tuning a pre-trained SITS-Former for solving an interested classification task. 
You can run the following Linux command to run the experiment:
```
python finetuning.py \
    --dataset_path '../s2-2017-IGARSS-NNI-NPY' \
    --pretrain_path '../checkpoints/pretrain/' \
    --finetune_path '../checkpoints/finetune/' \
    --num_workers 8 \
    --max_length 50 \
    --patch_size 5 \
    --num_features 10 \
    --num_classes 15 \
    --epochs 50 \
    --batch_size 128 \
    --hidden_size 256 \
    --layers 3 \
    --attn_heads 8 \
    --learning_rate 2e-4 \
    --weight_decay 1e-4 \
    --dropout 0.1
```

You can also run the following command to train SITS-Former from scratch for comparison:
```
python finetuning.py \
    --dataset_path '../s2-2017-IGARSS-NNI-NPY' \
    --pretrain_path None \
    --finetune_path '../checkpoints/finetune/' \
    --num_workers 8 \
    --max_length 50 \
    --patch_size 5 \
    --num_features 10 \
    --num_classes 15 \
    --epochs 300 \
    --batch_size 128 \
    --hidden_size 256 \
    --layers 3 \
    --attn_heads 8 \
    --learning_rate 2e-4 \
    --weight_decay 1e-4 \
    --dropout 0.1
	
```

## Experimental results
            
| Method | OA | Kappa | Medium F1-score |
| ------ | ---| ------| --------------- |
| Pretrained SITS-Former | 71.50 | 0.6120 | 43.57 |
| Non-pretrained SITS-Former | 61.95| 0.5154 | 39.92 |

