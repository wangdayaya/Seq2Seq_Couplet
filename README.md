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


## 案例
运行了 100 万个 batch，大约有 13 个 epoch ，看看使用 beamsearch 解码的效果，我觉得还阔以：

上联：海纳百川有容乃大
下联：天生万物无欲则刚

上联：我自横刀向天笑
下联：谁能放眼看云飞

上联：魑魅魍魉四小鬼
下联：鸳鸯鸳鸯一个人

上联：我寄愁心与明月
下联：谁知爱意比梅花

上联：少年不识愁滋味
下联：老子难忘苦感伤
