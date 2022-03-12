# -*- coding: utf-8 -*-
# @Time : 2022/3/12 18:24
# @Author : Jclian91
# @File : params.py
# @Place : Yangpu, Shanghai

# 模型参数
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-05
LABELS = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']
# 模型
MODEL_NAME_OR_PATH = './bert-base-uncased'
HIDDEN_LAYER_SIZE = 768
