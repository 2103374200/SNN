# Catching RNA’s Hiddens Marks: A Dual-Path Neural Network with Contrastive Learning for ac4C Prediction

N4-acetylcytidine (ac4C) is a significant RNA modification, particularly in the context of gene expression regulation and its implications in diseases. Laboratory-based identification of ac4C is costly and time-consuming, and existing deep learning models have shown unsatisfactory performance. Therefore, there is an urgent need to develop a highly accurate model for ac4C recognition. In response, we propose a Siamese network-based deep learning prediction model, named SNN-ac4C. Specifically, the model incorporates a dual-path contrastive learning architecture, leveraging BiLSTM and Multi-Head Self-Attention (MHSA) mechanisms to capture global contextual relationships, while employing Convolutional Neural Networks (CNN) for traditional biological feature extraction. The contrastive learning approach further enhances the model's ability to distinguish between ac4C-modified and unmodified sites. 

![model - 副本](https://github.com/user-attachments/assets/5de09389-6e29-4d08-a09a-771d8196bb22)

# Environment requirements
Before running, please make sure the following packages are installed in Python environment:

faiss_gpu==1.7.2

gensim==4.2.0

matplotlib==3.5.3

numpy==1.21.6

pandas==1.3.5

resneSt==0.0.5

scikit_learn==1.0.2

seaborn==0.13.2

shap==0.42.1

torch==1.13.1

torchvision==0.14.1

tqdm==4.65.2

transformers==4.30.2


# RUN
Changing working dir to SNN, python snn.py



