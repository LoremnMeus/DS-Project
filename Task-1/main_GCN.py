from rdkit import Chem
from rdchiral.template_extractor import extract_from_reaction
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import dgl
from torch import nn
from torch import optim
import pickle
import os
import torch
import traceback
import numpy as np

from model import GCN
from utils import gen_graph

DEVICE = "cuda:0"
FILE_TRAIN_DS = "train_dataset.pkl"
FILE_TEST_DS = "test_dataset.pkl"
FILE_TEMPLATES = "templates.pkl"
MODEL_SAVE_PATH = "model.ckpt"

# 数据预处理
if not os.path.exists(FILE_TRAIN_DS):
    train = pd.read_csv('raw_train.csv',sep=',',usecols=[2],header='infer').values[0::,0::]
    val = pd.read_csv('raw_val.csv',sep=',',usecols=[2],header='infer').values[0::,0::]
    test = pd.read_csv('raw_test.csv',sep=',',usecols=[2],header='infer').values[0::,0::]

    templates = {}
    train_dataset = []
    test_dataset = []
    i = 0

    for reaction in tqdm(train):
        reactants, products = reaction[0].split('>>')
        inputRec = {'_id':None,'reactants':reactants, 'products':products}
        smarts = extract_from_reaction(inputRec)
        if 'reaction_smarts' not in smarts:
            continue
        prod_templates = smarts['products'].split('.')
        template = smarts['reaction_smarts']
        if template not in templates:
            templates[template] = i
            i += 1
        label = templates[template]
        train_dataset.append((gen_graph(products), label))
        #for p in prod_templates:
        #    if p not in prod_templates_recations:
        #        prod_templates_recations[p] = (Chem.MolFromSmarts(p), [])
        #    prod_templates_recations[p][1].append(reaction)
    with open(FILE_TRAIN_DS, 'wb') as f:
        pickle.dump(train_dataset, f)

    for reaction in tqdm(test):
        reactants, products = reaction[0].split('>>')
        inputRec = {'_id':None,'reactants':reactants, 'products':products}
        smarts = extract_from_reaction(inputRec)
        if 'reaction_smarts' not in smarts:
            continue
        prod_templates = smarts['products'].split('.')
        template = smarts['reaction_smarts']
        if template not in templates:
            templates[template] = i
            i += 1
        label = templates[template]
        test_dataset.append((gen_graph(products), products, label))
        #for p in prod_templates:
        #    if p not in prod_templates_recations:
        #        prod_templates_recations[p] = (Chem.MolFromSmarts(p), [])
        #    prod_templates_recations[p][1].append(reaction)
    with open(FILE_TEST_DS, 'wb') as f:
        pickle.dump(test_dataset, f)

    with open(FILE_TEMPLATES, 'wb') as f:
        pickle.dump(templates, f)

else:
    with open(FILE_TRAIN_DS, 'rb') as f:
        train_dataset = pickle.load(f)
    with open(FILE_TEST_DS, 'rb') as f:
        test_dataset = pickle.load(f)
    with open(FILE_TEMPLATES, 'rb') as f:
        templates = pickle.load(f)

num_epoch = 120
batch_size = 2048
# 是否训练
train = False
# 是否加载上一次检查点
use_ckpt = False

#train_dataset = Dataset(train_dataset)
#train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

if train:
    model = GCN(31, 256, len(templates))
    if use_ckpt:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(DEVICE)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(num_epoch):
        batch_x = []
        batch_y = []
        for i, (x, y) in enumerate(tqdm(train_dataset)):
            if len(x.nodes()) == 1:
                continue
            batch_x.append(x.to(DEVICE))
            batch_y.append(y)
            if (i + 1) % batch_size == 0 or i == len(train_dataset):
                try:
                    output = model(dgl.batch(batch_x)).to("cpu")
                    loss = criterion(output, torch.tensor(batch_y))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                except Exception as e:
                    traceback.print_exc()
                batch_x = []
                batch_y = []


        print(epoch, loss.item())
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

#train_dataset = Dataset(train_dataset)
#test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 测试
model = GCN(31, 256, len(templates))
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(DEVICE)
model.eval()
i = 0
c = 0
k = 5   # Top-K
# 模板编号 : 模板产物
id_templates = {}
for template, idx in templates.items():
    id_templates[idx] = Chem.MolFromSmarts(template.split('>>')[1])
for x, x_mol, y in tqdm(test_dataset):
    try:
        # 获取输出
        output = model(x.to(DEVICE)).to("cpu").detach().numpy().squeeze()
        # 排序(从小到大，逆序读取)
        top = output.argsort()
        #for t in top[-k:]:
        #    if len(candidates) == k:
        #        break
        #    t = id_templates[t]
        #    if Chem.MolFromSmiles(x_mol).HasSubstructMatch(t):
        #        candidates.append(t)
        if y in top[-k:]:
            c += 1
        else:
            j = 0
            for t in top[::-1]:
                if j == k:
                    break
                # 检查目标产物是否存在模板产物的子结构
                if Chem.MolFromSmiles(x_mol).HasSubstructMatch(id_templates[t]):
                    if y == t:
                        c += 1
                        break
                j += 1
        i += 1
    except Exception as e:
        traceback.print_exc()

print(c / i)