import dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(__class__, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv4 = dglnn.GraphConv(hidden_dim, hidden_dim)
        #self.lstm = nn.LSTM(hidden_dim, n_classes)
        self.vertex_weight = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )
        self.classify = nn.Sequential(
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, g):
        # 应用图卷积和激活函数
        a = g.ndata['x']
        h = F.relu(self.conv1(g, a))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))
        w = dgl.apply_each(h, self.vertex_weight)
        with g.local_scope():
            g.ndata['h'] = h
            g.ndata['w'] = w
            # 使用平均读出计算图表示
            hg = dgl.mean_nodes(g, 'h', 'w')
            return self.classify(hg)
        
class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "gcn"))
        self.dropout = nn.Dropout(0.2)
        self.vertex_weight = nn.Sequential(
            nn.Linear(hid_size, 1),
            nn.Sigmoid() 
        )
        self.classify = nn.Sequential(
            nn.Linear(hid_size, out_size),
        )

    def forward(self, graph):
        x = graph.ndata['x'].float()
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        w = dgl.apply_each(h, self.vertex_weight)
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.ndata['w'] = w
            # 使用平均读出计算图表示
            hg = dgl.mean_nodes(graph, 'h', 'w')
            return self.classify(hg)
        
class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class GCNPlus(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(__class__, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv4 = dglnn.GraphConv(hidden_dim, hidden_dim)
        #self.conv5 = dglnn.GraphConv(hidden_dim, hidden_dim)
        #self.conv6 = dglnn.GraphConv(hidden_dim, hidden_dim)
        #self.conv7 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = MLPPredictor(hidden_dim, n_classes)

    def forward(self, g):
        # 应用图卷积和激活函数
        h = g.ndata['x']
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))
        #h = F.relu(self.conv5(g, h))
        #h = F.relu(self.conv6(g, h))
        #h = F.relu(self.conv7(g, h))
        h = self.classify(g, h)
        h = dgl.apply_each(h, F.sigmoid)
        return h
        #w = dgl.apply_each(h, self.vertex_weight)
        #w = dgl.apply_each(w, F.sigmoid)
        #with g.local_scope():
        #    g.ndata['h'] = h
        #    g.ndata['w'] = w
        #    # 使用平均读出计算图表示
        #    hg = dgl.mean_nodes(g, 'h', 'w')
        #    return self.classify(hg)