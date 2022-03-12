# -*- coding: utf-8 -*-
# @Time : 2022/3/12 18:25
# @Author : Jclian91
# @File : model_train.py
# @Place : Yangpu, Shanghai
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

import params
from model import BERTClass, loss_fn, save_ckp
from load_data import data_loader


val_targets = []
val_outputs = []


# 模型训练函数
def train_model(start_epochs, n_epochs, valid_loss_min_input, training_loader, validation_loader, model,
                optimizer, checkpoint_path, best_model_path):
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    for epoch in range(start_epochs, n_epochs + 1):
        train_loss = 0
        valid_loss = 0

        model.train()
        print('Epoch {}: Training Start'.format(epoch))
        for batch_idx, data in enumerate(training_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, BATCH： {batch_idx}, Training Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

        print('Epoch {}: Training End'.format(epoch))
        print('Epoch {}: Validation Start'.format(epoch))
        # validate the model
        model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                outputs = model(ids, mask, token_type_ids)

                loss = loss_fn(outputs, targets)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
                val_targets.extend(targets.cpu().detach().numpy().tolist())
                val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            print('Epoch {}: Validation End'.format(epoch))
            train_loss = train_loss / len(training_loader)
            valid_loss = valid_loss / len(validation_loader)
            # print training/validation statistics
            print('Epoch: {} \t Avgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'
                  .format(epoch, train_loss, valid_loss))

            # create checkpoint variable and add important data
            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': valid_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            # save checkpoint
            save_ckp(checkpoint, False, checkpoint_path, best_model_path)
            # save the model
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased from {:.6f} to {:.6f}). Saving model'
                      .format(valid_loss_min, valid_loss))
                # save checkpoint as best model
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                valid_loss_min = valid_loss

        print('Epoch {}  Done\n'.format(epoch))

    return model


if __name__ == '__main__':
    # 设置GPU或CPU训练
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available, '
              f'We will use the GPU: {torch.cuda.get_device_name(0)}.')
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    # 加载数据集
    training_loader, validation_loader = data_loader('./data/train.csv')
    # 创建模型
    model = BERTClass()
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.LEARNING_RATE)
    # 模型训练
    checkpoint_path = './models/current_checkpoint.pt'
    best_model = './models/best_model.pt'
    trained_model = train_model(1, params.EPOCHS, np.Inf, training_loader, validation_loader, model,
                                optimizer, checkpoint_path, best_model)
    # 模型预测指标
    val_predicts = (np.array(val_outputs) >= 0.5).astype(int)
    accuracy = accuracy_score(val_targets, val_predicts)
    f1_score_micro = f1_score(val_targets, val_predicts, average='micro')
    f1_score_macro = f1_score(val_targets, val_predicts, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    print(classification_report(val_targets, val_predicts))
