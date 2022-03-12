# -*- coding: utf-8 -*-
# @Time : 2022/3/12 18:29
# @Author : Jclian91
# @File : load_data.py
# @Place : Yangpu, Shanghai
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import params


# Torch数据集导入类
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.title = dataframe['TITLE']
        self.abstract = dataframe['ABSTRACT']
        self.targets = self.data.target_list
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        abstract = str(self.abstract[index])

        inputs = self.tokenizer.encode_plus(
            title,
            abstract,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation='second_only'
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(self.targets[index], dtype=torch.float)
               }


def data_loader(raw_csv_path):
    df_raw = pd.read_csv(raw_csv_path)
    df_raw['target_list'] = df_raw[params.LABELS].values.tolist()
    df = df_raw[['TITLE', 'ABSTRACT', 'target_list']].copy()
    # 划分数据, 训练集:测试集=8:2
    train_size = 0.8
    train_dataset = df.sample(frac=train_size, random_state=200)
    valid_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    print(f"FULL Dataset: {df.shape}, "
          f"TRAIN Dataset: {train_dataset.shape}, "
          f"TEST Dataset: {valid_dataset.shape}")
    # 使用改造后的Torch DataLoader加载数据集
    tokenizer = BertTokenizer.from_pretrained(params.MODEL_NAME_OR_PATH)
    training_set = CustomDataset(train_dataset, tokenizer, params.MAX_LEN)
    validation_set = CustomDataset(valid_dataset, tokenizer, params.MAX_LEN)
    train_params = {'batch_size': params.TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    test_params = {'batch_size': params.VALID_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}
    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **test_params)
    return training_loader, validation_loader
