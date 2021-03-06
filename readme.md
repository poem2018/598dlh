# Repo for 598DLH final project

This is the repo for 598DLH final project.

- Ruike Zhu
- Pan Liu

## Problem 

We reproduce the Sherbet model, link of paper: https://arxiv.org/pdf/2106.04751.pdf 

## Model structure

![results](https://github.com/poem2018/598dlh/blob/main/pics/model_pic.png)

## Implementation

We have done the experiments in the below table

![results](https://github.com/poem2018/598dlh/blob/main/pics/train_table.png)

## Reproduced Results

### MIMIC-III
    for Heart Failure task:

    |               | F1                 | AUC score            | 
    | ------------  | -----------------  | ------------------   | 
    | dropout = 0.2 | 0.7307171853856563 | 0.8612635647903025   |
    | dropout = 0.4 | 0.7385019710906703 | 0.8666701103262437   |
    | dropout = 0.6 | 0.7409470752089138 |  0.8633082376641656  |
    | dropout = 0.8 | 0.7340720221606648 | 0.8634502886217184   |
    | sherbet_b     | 0.744              | 0.8657532359638589   |

    for diagnosis task
    |             | recall@10         | recall@20          |  F1                    |
    | ------------| ----------------- | ------------------ | ---------------------- |
    | sherbet_a   | 0.3920123         | 0.40286424         |   0.2410660363641088   |
    | sherbet_b   | 0.3884631         | 0.40051201         |   0.23019432399405035  |

### eICU dataset
    for diagnosis task
    |              | recall@10         | recall@20          |  F1                   |
    | ------------ | ----------------- | ------------------ | --------------------- |
    | sherbet_a    | 0.78273214        | 0.82769583         | 0.6155913694704069    |

*sherbet_a: follow the origin parameters provided in paper: Sherbet with self-supervised learning and hierarchical prediction, and pretrain with hyperbolic embedding. 

*sherbet_b: removing hyperbolic embedding part in pretrain


### visualize
We visualize our training result in pictures like below, all the score is get from validation data.
Please refer to the *output_res* folder if you want to see other visualized results

![results](https://github.com/poem2018/598dlh/blob/main/output_res/mimic_hf_self_hi_hy_08.png)

## How to run 
```
python data_preprocess.py
python run_hyperbolic_embedding.py
python main.py
```


## Citation
```
Lu, Chang & Reddy, Chandan & Ning, Yue. (2021). Self-Supervised Graph Learning With Hyperbolic Embedding for Temporal Health Event Prediction. IEEE Transactions on Cybernetics. PP. 1-13. 10.1109/TCYB.2021.3109881. 
```