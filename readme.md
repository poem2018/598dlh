# Repo for 598DLH final project

This is the repo for 598DLH final project.

- Ruike Zhu
- Pan Liu

## Problem 

We reproduce the Sherbet model, link of paper: https://arxiv.org/pdf/2106.04751.pdf 

## Model structure

![results](https://github.com/poem2018/598dlh/blob/main/pics/model_pic.png)

## Results

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
    | sherbet_a    | 1                 | 3                  |                       |




