# Temporal-Enhanced DeepEmoCluster

:exclamation::exclamation::exclamation:**semi-supervised model implementation is on the way**

The [temporal-enhanced DeepEmoCluster](https://doi.org/10.1016/j.specom.2023.103027) adds additional sentence-level temporal modeling to further improve the [DeepEmoCluster](https://github.com/winston-lin-wei-cheng/DeepEmoClusters) recognition performances. It uses the two proposed temporal modeling approaches:
1. Temporal Net- *Temp-GRU*, *Temp-CNN*, *Temp-Trans*
2. Triplet loss- *Temp-Triplet*


NOTE: The experiments and the provided pretrained models were based on the MSP-Podcast v1.8 corpus in the paper.

![The Temporal-Enhanced DeepEmoCluster Framework](/images/XXX.png)


# Suggested Environment and Requirements
1. Python 3.6+
2. Ubuntu 18.04
3. CUDA 10.0+
4. pytorch version 1.4.0
5. librosa version 0.7.0
6. faiss version 1.6.0
7. The scipy, numpy and pandas...etc common packages
8. The MSP-Podcast corpus (request to download from [UTD-MSP lab website](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html))


# Feature Extraction & Preparation
Using the **feat_extract.py** to extract 128-mel spectrogram features for every speech segment in the corpus (remember to change I/O paths in the .py file). Then, use the **norm_para.py** to save normalization parameters for our framework's pre-processing step. The parameters will be saved in the generated *'NormTerm'* folder. We have provided the parameters of the v1.8 corpus in this repo.


# Training from Scratch
1. Prepare the 128-mel spectrogram features of the MSP-Podcast corpus (it can be any version)
2. Change data root & label paths (the *'labels_concensus.csv'* file provided with the corpus) in **main.py**, the running args are,
   * -ep: number of epochs
   * -batch: batch size for training
   * -emo: emotional attributes (Act, Dom or Val)
   * -nc: number of clusters in the latent space for the cluster classifier
   * -mt: temporal modeling type (Temp-GRU, Temp-CNN, Temp-Trans or Temp-Triplet)
   * run in the terminal
   * the trained models will be saved under the generated *'trained_models'* folder
```
python main.py -ep 30 -batch 64 -emo Val -nc 10 -mt Temp-Triplet
```
3. Evaluation for the trained models using the **online_testing.py**. The results are based on the MSP-Podcast pre-defined test set,
   * run in the terminal
```
python online_testing.py -ep 30 -batch 64 -emo Val -nc 10 -mt Temp-Triplet
```

# Pre-trained models
We provide the pretrained models based on **version 1.8** of the MSP-Podcast in the *'trained_models'* folder. The CCC performances of models based on the test set are shown in the following table. Note that the results are slightly different from the [paper](https://doi.org/10.1016/j.specom.2023.103027) since we performed statistical test in the paper (i.e., we averaged multiple trails).

| Temporal Modeling Approach | Act(10-clusters) | Dom(10-clusters) | Val(10-clusters) |
|:----------------:|:----------------:|:----------------:|:----------------:|
| Temp-GRU | 0.5730 | 0.4663 | 0.1590 |
| Temp-CNN | 0.5630 | 0.4728 | 0.1471 |
| Temp-Trans | 0.5672 | 0.4632 | 0.1712 |
| Temp-Triplet | 0.5664 | 0.4774 | 0.1648 |

Users can get these results by running the **online_testing.py** with the corresponding args.


# End-to-End Emotion Prediction Process
We provide the end-to-end prediction process that alows users to directly make emotion predictions (i.e., arousal, domiance and valence) on your own dataset or any audio files (audio spec: WAV file, 16k sampling rate and mono channel) based on the provided pretrained models. Users just need to change the input folder path in **prediction_process.py** to run the predictions and the output results will be saved as a *'pred_result.csv'* file under the same directory. 


# Reference
If you use this code, please cite the following paper:

Wei-Cheng Lin and Carlos Busso, "Deep temporal clustering features for speech emotion recognition"

```
@article{Lin_2024,
  author = {W.-C. Lin and C. Busso},
  title = {Deep temporal clustering features for speech emotion recognition},
  journal = {Speech Communication},
  volume = {157},
  number = {},
  year = {2024},
  pages = {103027},
  month = {February},
  doi={10.1016/j.specom.2023.103027},
} 
```
