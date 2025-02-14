<div align='center'>
<h1>Language Representation Favored Zero-Shot Cross-Domain Cognitive Diagnosis, KDD 2025</h1>
<a href='https://aiedu.ecnu.edu.cn/'>Shanghai Institute of AI Education</a>, <a href='http://www.cs.ecnu.edu.cn/'>School of Computer Science and Technology</a>,  and <a href='https://ed.ecnu.edu.cn/'> <br/>Faculty of Education</a>, East China Normal University

<br/>
<br/>
<a href='https://github.com/ECNU-ILOG/LRCD'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://github.com/ECNU-ILOG/LRCD/tree/main/paper/Language Representation Favored Zero-Shot Cross-Domain Cognitive Diagnosis.pdf'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a>




<br/>
<img src="img/Framework.svg" width="800" alt="Framework Image" />

</div>

------



:smile_cat: Welcome to LRCD, this is a comprehensive repository specializing in ***Language Representation Favored Zero-Shot Cross-Domain Cognitive Diagnosis*** published in KDD 2025.


> Here, we propose LRCD, a new paradigm: a cognitive diagnosis framework that solely relies on language representation. It is both model-agnostic and scenario-agnostic.


# Introduction of LRCD: Challenge, Solution and Insights

## Challenge

Cognitive diagnosis aims to infer students' mastery levels based on their historical response logs. However, existing cognitive diagnosis models (CDMs), **which rely on ID embeddings, often have to train specific models on specific domains. This limitation may hinder their directly practical application in various target domains, such as different subjects (e.g., Math, English and Physics) or different education platforms (e.g., ASSISTments, Junyi Academy and Khan Academy).**


<div align='center'>
    <img src="img/ZSCD.svg" width="250" alt="Framework Image">
</div>


## Solution

To address this issue, this paper proposes language representation favored zero-shot cross-domain cognitive diagnosis (LRCD). Specifically, LRCD first **analyzes the behavior patterns of students, exercise and concepts in different domains, and then describe the profiles of students, exercises and concepts using textual descriptions (TCP).** Via recent advanced text-embedding modules, these profiles can be transformed to vectors in the unified language space. Moreover, to address the discrepancy between the language space and the cognitive diagnosis space, we propose **language-cognitive mappers (LCM) in LRCD to learn the mapping from the former to the latter**. Then, these profiles can be simply and efficiently integrated and trained with existing CDMs.

<div align='center'>
  <img src="img/TCP.png" alt="TCP" width="300"/>
  <img src="img/LCM.png" alt="LCM" width="300"/>
</div>

## Insights

LRCD is not only **model-agnostic but also scenario-agnostic**, meaning it can be applied across various contexts, including Transductive CD, Inductive CD, Zero-Shot CD and Computerized Adaptive Testing. 

### Zero-Shot CD (Same Platform, Different Subjects)
<div align='center'>
<img src="img/Table-1.png"  width="600"/>
</div>

### Zero-Shot CD (Different Platform, Same Subjects)
<div align='center'>
<img src="img/Table-2.png"  width="600"/>
</div>

### Overlap CD (Same Platform, Different Subjects, Overlap Students)

<div align='center'>
<img src="img/overlap.png"  width="300"/>
</div>

### Transductive CD (Standard Scenarios)

<div align='center'>
<img src="img/standard.png"  width="600"/>
</div>


## 📰 News 
- [x] [2024.12.16] Upload the introduction.
- [x] [2024.12.6] LRCD v1.0 is released.

# Requirements	

```
joblib==1.3.2
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.2
scipy==1.10.1
torch==2.1.1
wandb==0.16.2
```
Please install all the dependencies listed in the `requirements.txt` file by running the following command:

```bash
pip install -r requirements.txt
```

# Data Preprocess

You should process datasets by yourself, you need first 

> cd data



Noting: Due to some embedding text files being too large, we have zipped them. Before starting to run, you need to unzip all files. We upload the zip file on Google Drive. https://drive.google.com/file/d/10A8fdmqLXlMyw824_1zj0btaiRQ07mhf/view?usp=drive_link

# Experiments

Then, you can choose different diagnostic methods based on the provided dataset to run this code. Here is an example:

```shell
python main.py --method=orcdf --train_file=data/SLP-BIO,data/SLP-PHY --test_file=data/SLP-MAT --seed=0 --batch_size=256 --device=cuda:0 --epoch_num=20  --lr=2.5e-4 --latent_dim=64
```


# Reference :thought_balloon:

Shuo Liu, Zihan Zhou, Yuanhao Liu, Jing Zhang, Hong Qian "Language Representation Favored Zero-Shot Cross-Domain Cognitive Diagnosis." In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2025.



## Bibtex

```
@inproceedings{Liu2025LRCD,
author = {Shuo Liu and Zihan Zhou and Yuanhao Liu and Jing Zhang and Hong Qian},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
title = {Language Representation Favored Zero-Shot Cross-Domain Cognitive Diagnosis},
year = {2025},
address={Toronto, Canada}
}
```
 
