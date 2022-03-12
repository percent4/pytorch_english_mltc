# -*- coding: utf-8 -*-
# @Time : 2022/3/12 18:36
# @Author : Jclian91
# @File : model.py
# @Place : Yangpu, Shanghai
import shutil
import torch
import transformers

from params import MODEL_NAME_OR_PATH,  HIDDEN_LAYER_SIZE


# 模型类
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(MODEL_NAME_OR_PATH)
        self.l2 = torch.nn.Dropout(0.2)
        self.l3 = torch.nn.Linear(HIDDEN_LAYER_SIZE, 6)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


# 定义损失函数
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


# 保存模型
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_path = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_path)
