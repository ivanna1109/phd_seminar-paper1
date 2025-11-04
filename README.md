# PhD Studies - Subject: Seminar Paper 1
## Topic - Evaluation of Self-Supervised Learning Approach and Deep Convolutional Networks for Classification of Human Embryo Images

## Overview

This repository documents an in-depth experimental study focused on applying **Self-Supervised Learning (SSL)** and **Convolutional Neural Networks (CNNs)** to a critical problem in biomedicine: the **Classification of Human Embryo Images**.

The primary goal is to **develop robust AI models** capable of accurately differentiating between high and low potential embryos, specifically addressing the challenges posed by **data imbalance** inherent in medical image analysis.
The project involves **comparative analysis of several architectures** (including specialized models like EmbryoNet and transfer learning candidates like ResNet18 and MobileNetV3) and the testing of **various regularization and data balancing techniques** to find the optimal strategy for maximizing performance on the minority class.

### Repository Contents
  * **`localWork/`**
    * **`data_processing/`**
        * Contains scripts responsible for **processing raw data**, **(EDA)**, data transformation and visualization, and creating the final parquet files for train and test sets
    * **`imgs/`** - plots from EDA file
  * **`models/`** - definition of used models
  * **`pretraining/`** - dirs and files related to pretraining task of SSL
  * **`training/`** - dirs and files related to training process of every model
  * **`training_evaluation/`** - dirs and filers related to evaluation process, after training is done
  * **`visualization_results/`** - files related to results obtained from training process
   
      
