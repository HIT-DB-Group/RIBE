import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
import torch
import optuna
from sklearn.metrics import fbeta_score
from selection.utils import *

from ChangeFormer.database_util import collator
from ChangeFormer.dataset import *
from ChangeFormer.model import ChangeFormer

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def evaluate(model, ds, bs, device):
    model.eval()
    y_true=torch.FloatTensor([])
    y_pred=torch.FloatTensor([])

    with torch.no_grad():
        for i in range(0, len(ds), bs):
            batch, batch_labels = collator(list(zip(*[ds[j] for j in range(i,min(i+bs, len(ds)) ) ])))
            batch_labels = batch_labels.to(torch.float32)
            _, true_labels = torch.max(batch_labels.data, 1)
            y_true=torch.cat((y_true,true_labels))
            batch = batch.to(device)

            change_preds = model(batch)
            change_preds = change_preds.squeeze()
            preds=change_preds.to('cpu')
            _, pred_labels = torch.max(preds.data, 1)
            y_pred=torch.cat((y_pred,pred_labels))
    accuracy = (y_pred == y_true).sum().item() / len(y_true)
    f2=fbeta_score(y_true, y_pred, beta=2)
    return f2,accuracy

def train(model, train_ds, val_ds, crit, \
    config, optimizer=None, scheduler=None):
    
    bs, device, epochs, clip_size = \
        config['bs'],config['device'],config['epochs'], config['clip_size']
    lr = config['lr']

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.7)
    t0 = time.time()
    rng = np.random.default_rng()
    best_accuracy = 0

    for epoch in range(epochs):
        losses = 0
        y_true=torch.FloatTensor([])
        y_pred=torch.FloatTensor([])
        model.train()
        train_idxs = rng.permutation(len(train_ds))
        
        for idxs in chunks(train_idxs, bs):
            optimizer.zero_grad()
            batch, batch_labels = collator(list(zip(*[train_ds[j] for j in idxs])))
            _, true_labels = torch.max(batch_labels.data, 1)
            y_true=torch.cat((y_true,true_labels))
            batch_change_label = batch_labels.to(torch.float32).to(device)
            batch = batch.to(device)
            
            change_preds = model(batch)
            change_preds = change_preds.squeeze()

            loss = crit(change_preds, batch_change_label)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_size)

            optimizer.step()
            losses += loss.item()
            preds=change_preds.to('cpu')
            _, pred_labels = torch.max(preds.data, 1)
            y_pred=torch.cat((y_pred,pred_labels))
        accuracy = (y_pred == y_true).sum().item() / len(y_true)
        f2=fbeta_score(y_true, y_pred, beta=2)
        test_f2 ,test_accuracy= evaluate(model, val_ds, bs, device)
        if (epoch+1) % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {losses/len(train_idxs)}, Train F2 Score: {f2}, Accuracy {accuracy}, Test F2 Score: {test_f2}, Accuracy {test_accuracy}')
        if test_f2 > best_accuracy:
            best_accuracy = test_f2
    return model, best_accuracy


def logging(args, epoch, test_accuracy, filename = None, save_model = False, model = None):
    arg_keys = [attr for attr in dir(args) if not attr.startswith('__')]
    arg_vals = [getattr(args, attr) for attr in arg_keys]
    
    res = dict(zip(arg_keys, arg_vals))
    model_checkpoint = str(hash(tuple(arg_vals))) + '.pt'

    res['epoch'] = epoch
    res['model'] = model_checkpoint 
    res['accuracy']=test_accuracy

    filename = args.newpath + filename
    model_checkpoint = args.newpath + model_checkpoint
    
    if filename is not None:
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            df = df.append(res, ignore_index=True)
            df.to_csv(filename, index=False)
        else:
            df = pd.DataFrame(res, index=[0])
            df.to_csv(filename, index=False)
    if save_model:
        torch.save({
            'model': model.state_dict(),
            'args' : args
        }, model_checkpoint)
    
    return res['model']  

if __name__ == '__main__':
    cache=load_checkpoint(f'./ChangeFormer/dataset/example.cache')
    collated_path=f'./ChangeFormer/dataset/data.pkl'
    ptd=PlanChangeDataset(cache)
    ptd.generate_data()
    ptd.shuffle()
    ptd.dump_collated_dicts(collated_path)

    config={'lr': 0.0005789346253890466,
        'bs': 256,
        'epochs': 30,
        'clip_size': 8,
        'embed_size': 64,
        'pred_hid': 128,
        'ffn_dim': 256,
        'head_size': 16,
        'n_layers': 1,
        'join_schema_head_size': 2,
        'attention_dropout_rate': 0.06414118211287106,
        'join_schema_layers': 4,
        'dropout': 0.07326263251335534,
        'sch_decay': 0.22424663955914978,
        'device':'cuda:0'}

    train_ds,val_ds = get_ts_vs(collated_path,0.7)
    model = ChangeFormer(emb_size=config['embed_size'], ffn_dim=config['ffn_dim'], \
                head_size=config['head_size'], join_schema_head_size=config['join_schema_head_size'], \
                dropout=config['dropout'],attention_dropout_rate=config['attention_dropout_rate'], \
                n_layers=config['n_layers'],join_schema_layers=config['join_schema_layers'],pred_hid=config['pred_hid'])
    _ = model.to(config['device'])
    position_weight = torch.tensor([1.0, 2.0]).to(config['device'])
    crit = nn.BCEWithLogitsLoss(pos_weight=position_weight)
    model,accuracy = train(model, train_ds, val_ds, crit, config)
    
    torch.save(model.state_dict(), f"./ChangeFormer/dataset/model.pth")


