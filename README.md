# Seq2Seq 写对联


## 项目说明
Tensorflow 1.14 实现 Seq2Seq ，并训练实现对对联

## Seq2Seq 架构

方法一、使用简单的架构，在 model.py 文件中实现

方法二、使用 Attention 和 TrainingHelper 训练模型，然后使用 BeamSearch 预测结果，在 model_att_beamsearch.py 中实现

## Data

在 data 目录中有 70w+ 对的对联数据


## Train

执行：python trian.py

## test

执行：python test.py 



