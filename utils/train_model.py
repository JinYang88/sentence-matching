import time
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def predict_on(model, data_dl ,model_state_path=None):
    model.eval()
    save_list = []
    res_list = []
    label_list = []
    best_acc = -1
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for batch_data in data_dl:
        label = batch_data[-1]
        y_pred = model(batch_data).sigmoid()
        save_list.extend(y_pred)
        label_list.extend(label)
        
    for threshold in thresholds:
        res_list = list(map(lambda x: 1 if x >= threshold else 0, save_list))
        Acc = accuracy_score(res_list, label_list)
        Precision = precision_score(res_list, label_list)
        Recall = recall_score(res_list, label_list)
        F1 = f1_score(res_list, label_list)
        if best_acc < Acc:
            best_th, best_acc, best_Precision, best_Recall, best_F1 = threshold, acc, Precision, Recall, F1
            best_acc = Acc

    # with open("{}_prob.txt".format(model.__class__.__name__), 'w') as fw:
    #     for item in save_list:
    #         fw.write('{}\n'.format(item))
    
    return best_th, best_acc, best_Precision, best_Recall, best_F1

def train(model, train_iter, valid_iter,
                checkpoint_path, device, epochs=10,
                print_every=100, early_stop_num=25):
    print('Start training model [{}].'.format(model.__class__.__name__))
    best_metric = 0
    worse_count = 0
    stop = False
    best_info = {}
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(parameters, lr=1e-3)
    loss_func = nn.BCEWithLogitsLoss()
    batch_num = train_iter.batch_num
    batch_start = time.time()
    for i in range(epochs) :
        train_iter.shuffle()
        batch_count = 0
        for batch_data in train_iter:
            if len(batch_data[0]) == 1: continue
            label = torch.tensor(batch_data[-1]).float().to(device)
            model.train()
            y_pred = model(batch_data)
            weight = torch.Tensor(list(map(lambda x: 0.82 if x==1 else 0.18, label.view(-1)))).to(device)
            loss_func = nn.BCEWithLogitsLoss(weight=weight)
            loss = loss_func(y_pred.view(-1), label.view(-1))
            model.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1
            if batch_count % print_every == 0:
                print("------")
                th, Acc, Precision, Recall, F1 = predict_on(model, valid_iter)
                batch_end = time.time()
                if Acc >= best_metric:
                    worse_count = 0
                    best_metric = Acc
                    best_info["Acc"] = Acc
                    best_info["Epoch"] = i + 1
                    best_info['checkpoint_path'] = checkpoint_path
                    best_info['th'] = th
                    print("Saving model..")
                    torch.save(model.state_dict(), checkpoint_path)           
                else:
                    worse_count += 1
                    if worse_count > early_stop_num:
                        print("Early stop at epoch [{}], best f1 [{:.3f}].".format(i+1, best_metric))
                        stop = True
                        break
                print('Batch:[{}/{}], Epoch:[{}/{}], Time:[{:.3f}s].'.format(batch_count, batch_num, i+1, epochs, round(batch_end - batch_start, 2)))
                print('th:[{}], Acc:[{:.3f}], Precision:[{:.3f}], Recall:[{:.3f}], Best Acc:[{:.3f}]'
                      .format(th, Acc, Precision, Recall, best_metric))
        if stop:
            break
    return best_info
