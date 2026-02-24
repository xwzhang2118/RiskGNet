import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, HANConv
from sklearn.model_selection import KFold
from sklearn.metrics import auc, average_precision_score, f1_score, roc_auc_score, accuracy_score
from sklearn.neighbors import kneighbors_graph


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='STAD', help='Device to use')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--in_dim', type=int, default=128, help='Number of input features')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Number of hidden units')
    parser.add_argument('--out_dim', type=int, default=128, help='Number of output units')
    
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GCN layers')
    parser.add_argument('--learning_rate', type=float, default=0.007, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Learning rate')
    parser.add_argument('--t', type=float, default=0.1, help='Device to use')
    parser.add_argument('--k', type=int, default=3, help='Device to use')
    args = parser.parse_args()
    return args

def load_data(args, device):
    # load test data
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    test_edge = pd.read_csv(current_file_path + '/' + args.dataset +'/test_edge_name.txt', sep=' ', header=None).values
    test_edge_index = pd.read_table(current_file_path + '/' +args.dataset + "/test_edge_index.txt", sep=' ', header=None).values
    node_feature = pd.read_csv(current_file_path + '/' + args.dataset + '/node_feature.txt', sep=' ', header=None).values
    snp = pd.read_table(current_file_path + '/' + args.dataset + "/snp_list.txt", sep='\t').values[:, 0]
    
    # load real graph
    adj_real = pd.read_table(current_file_path + '/' + args.dataset + '/adj_real_2.txt', sep=' ', header=None).values
    snp_gene_real = np.where(np.triu(adj_real) == 1)
    data_real = torch.load(current_file_path + '/' + args.dataset + '/data_2.pth')

    print(data_real)

    #load sequence graph
    k = args.k
    feature_str = pd.read_table(current_file_path + '/' + args.dataset + '/feature_str.txt', sep=' ', header=None).values
    adj_str= kneighbors_graph(feature_str, k, mode='connectivity', include_self=False).toarray()
    str_inter = np.where(np.triu(adj_str) == 1)
    edges = np.vstack(str_inter).T
    snp_inter_str = edges[(edges[:, 0] < len(snp)) & (edges[:, 1] < len(snp))]
    snp_gene = edges[(edges[:, 0] < len(snp)) & (edges[:, 1] >= len(snp))]
    snp_gene_str = np.vstack((snp_gene[:, 0], snp_gene[:, 1] - len(snp))).T
    gene_inter = edges[(edges[:, 0] >= len(snp)) & (edges[:, 1] >= len(snp))]
    gene_inter_str = np.vstack((gene_inter[:, 0] - len(snp), gene_inter[:, 1] - len(snp))).T

    data_str = HeteroData()
    data_str['snp'].x = torch.tensor(node_feature[:len(snp)], dtype=torch.float)
    data_str['gene'].x = torch.tensor(node_feature[len(snp):], dtype=torch.float)
    data_str['snp', 'to', 'snp'].edge_index = torch.tensor(snp_inter_str, dtype=torch.long).t().contiguous()
    data_str['snp', 'to', 'gene'].edge_index = torch.tensor(snp_gene_str, dtype=torch.long).t().contiguous()
    data_str['gene', 'to', 'gene'].edge_index = torch.tensor(gene_inter_str, dtype=torch.long).t().contiguous()
    print(data_str)

    snp_gene_real = torch.tensor(snp_gene_real, dtype=torch.long).contiguous().to(device)
    str_inter = torch.tensor(str_inter, dtype=torch.long).contiguous().to(device)
    adj_str = adj_str + np.eye(adj_str.shape[0])
    adj_str = torch.tensor(adj_str, dtype=torch.float).to(device)
    node_feature = torch.tensor(node_feature, dtype=torch.float).to(device)
    feature_str = torch.tensor(feature_str, dtype=torch.float).to(device)
    
    return  data_real, snp_gene_real, node_feature, adj_real, data_str, str_inter, feature_str, adj_str, test_edge_index, test_edge, len(snp)
    

def get_positive_negative_samples(adj_real, test, lenth):
    positive_samples = np.array(np.where(np.triu(adj_real == 1)))
    negative_samples = np.array(np.where(np.triu(adj_real == 0)))
    condition_pos = np.where((positive_samples[0] < lenth ) & (positive_samples[1] >= lenth))
    condition_neg = np.where((negative_samples[0] < lenth ) & (negative_samples[1] >= lenth))
    positive_samples = positive_samples[:, condition_pos[0]]
    negative_samples = negative_samples[:, condition_neg[0]]
    
    test_set = set(tuple(pair) for pair in test.T)
    positive_samples_filtered = []
    for i in range(positive_samples.shape[1]):
        pair = (positive_samples[0, i], positive_samples[1, i])
        if pair not in test_set:
            positive_samples_filtered.append(pair)
    positive_samples = np.array(positive_samples_filtered).T

    negative_samples_filtered = []
    for i in range(negative_samples.shape[1]):
        pair = (negative_samples[0, i], negative_samples[1, i])
        if pair not in test_set:
            negative_samples_filtered.append(pair)
    
    negative_samples_filtered = np.array(negative_samples_filtered).T
    if negative_samples_filtered.size == 0:
        raise ValueError("No valid negative samples found after filtering out test set.")
    
    num_positive_samples = positive_samples.shape[1]
    negative_sample_indices = np.random.choice(negative_samples_filtered.shape[1], num_positive_samples*2, replace=True)
    negative_samples = negative_samples_filtered[:, negative_sample_indices]

    samples = np.hstack([positive_samples, negative_samples]).T
    labels = np.hstack([np.ones(num_positive_samples), np.zeros(num_positive_samples)])
    
    shuffle_indices = np.random.permutation(len(labels))
    samples = samples[shuffle_indices]
    labels = labels[shuffle_indices]
    
    samples = torch.tensor(samples, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)
    
    return samples, labels


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x1 = F.relu(self.gcn1(x, edge_index))
        x2 = self.gcn2(x1, edge_index)
        return x2

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class EnhancedHAN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, data, num_layers, heads, dropout):
        super(EnhancedHAN, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            input_dim = in_channels if i == 0 else hidden_channels
            self.convs.append(HANConv(input_dim, hidden_channels, data.metadata(), heads=heads))
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fusion_weights = nn.Parameter(torch.ones(num_layers))
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        layer_outputs = []
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            for node_type in x_dict:
                x_dict[node_type] = self.norms[i](x_dict[node_type])
                x_dict[node_type] = self.dropout(x_dict[node_type])
            layer_outputs.append(x_dict)
        fused_x_dict = {}
        for node_type in x_dict:
            fused_x = sum([self.fusion_weights[i] * layer[node_type] for i, layer in enumerate(layer_outputs)])
            fused_x /= self.fusion_weights.sum()
            fused_x_dict[node_type] = fused_x

        z = torch.cat([fused_x_dict['snp'], fused_x_dict['gene']], dim=0)
        z = self.feature_projection(z)
        out = self.out_layer(z)
        return out

class RiskGNet(nn.Module):
    def __init__(self,data, in_dim, hidden_dim, out_dim, num_heads, num_layers, dropout):
        super(RiskGNet, self).__init__()
        self.CGCN = GCN(in_dim, hidden_dim, out_dim, dropout)
        self.real_graph = EnhancedHAN(in_dim, hidden_dim, out_dim, data_real, num_layers, num_heads, dropout)
        self.feature_graph = EnhancedHAN(in_dim, hidden_dim, out_dim, data_str, num_layers, num_heads, dropout)
        self.attention = Attention(in_dim)
        self.output_proj = nn.Linear(3*in_dim, out_dim)

    def forward(self, data_real, snp_gene_real, node_feature, adj_real, data_str, str_inter, feature_str, adj_feature):
        # Personalized
        emd_real = self.real_graph(data_real)

        # Shared
        com1 = self.CGCN(node_feature, snp_gene_real)
        com2 = self.CGCN(node_feature, str_inter)
        Xcom = (com1+com2)/2

        # Personalized
        emd_str = self.feature_graph(data_str)

        combined_x = torch.stack([emd_real, Xcom, emd_str], dim=1)
        emb, att = self.attention(combined_x)
        output = emb
        return output, com1, com2, emd_real, emd_str, att

class LinkPredictorWithContrastiveLearning(nn.Module):
    def __init__(self, model,input_dim, hidden_dim, temperature):
        super(LinkPredictorWithContrastiveLearning, self).__init__()
        self.model = model
        self.classifier = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.temperature = temperature
    
    def forward(self, data_real, snp_gene_real, node_feature, adj_real, data_str, str_inter, feature_str, adj_feature, edge_index):
        # emd
        emd_all, com1, com2, emd_str, emd_real, att = self.model(data_real, snp_gene_real, node_feature, adj_real, data_str, str_inter, feature_str, adj_feature)
        
        # link prediction
        edge_embeds = torch.cat(
            [emd_all[edge_index[:, 0]], emd_all[edge_index[:, 1]]], dim=-1
        )

        link_predictions = torch.sigmoid(self.classifier(edge_embeds))
        

        return link_predictions,emd_all, com1, com2, emd_str, emd_real, att

    def contrastive_loss(self, embeddings_real, embeddings_str):
        embeddings_real = F.normalize(embeddings_real, dim=1)
        embeddings_str = F.normalize(embeddings_str, dim=1)
        similarity_matrix = torch.mm(embeddings_real, embeddings_str.T) / self.temperature
        labels = torch.arange(embeddings_real.size(0)).to(similarity_matrix.device)
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss
    


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = args_parser()
    data_real, snp_gene_real, node_feature, adj_real, data_str, str_inter, feature_str, adj_feature, test_edge_index, test_edge, len_snp = load_data(args, device)
    samples, labels = get_positive_negative_samples(adj_real, test_edge_index, len_snp)
    samples, labels = np.array(samples), np.array(labels)

    # 5-cv
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_fold_auc = []
    all_fold_aupr = []
    all_fold_f1 = []
    all_fold_acc = []

    # parameters
    in_dim = node_feature.shape[1]
    hidden_dim = args.hidden_dim
    out_dim = args.out_dim
    num_heads = args.num_heads
    num_layers = args.num_layers
    dropout = args.dropout
    temperature = args.temperature
    epochs = args.epochs
    learning_rate = args.learning_rate
    patience = args.patience
    t = args.t
    k = args.k

    best_val_loss = float('inf')
    trigger_times = 0

    adj_real = adj_real + np.eye(adj_real.shape[0])
    adj_real = torch.tensor(adj_real, dtype=torch.float).to(device)

    for fold, (train_index, val_index) in enumerate(kf.split(samples)):
        print(f"\nFold {fold + 1}")

        train_samples, val_samples = samples[train_index], samples[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]
        
        data_real = data_real.to(device)
        data_str = data_str.to(device)
        edge_index_train = torch.tensor(train_samples, dtype=torch.long).to(device)
        labels_train = torch.tensor(train_labels, dtype=torch.float).to(device)
        edge_index_val = torch.tensor(val_samples, dtype=torch.long).to(device)
        labels_val = torch.tensor(val_labels, dtype=torch.float).to(device)
        
        #model
        dual_graph_transformer = RiskGNet(data_real, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                                                    num_heads=num_heads, num_layers=num_layers, dropout=dropout).to(device)
        link_predictor = LinkPredictorWithContrastiveLearning(dual_graph_transformer, input_dim=in_dim, hidden_dim=hidden_dim, temperature=temperature).to(device)
        
        optimizer = torch.optim.Adam(link_predictor.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        best_val_loss = float('inf')
        trigger_times = 0
        # training
        for epoch in range(epochs):
            link_predictor.train()
            optimizer.zero_grad()
            
            link_predictions, emd, com1, com2, emd_str, emd_real,att = link_predictor(data_real, snp_gene_real, node_feature, adj_real, data_str, str_inter, feature_str, adj_feature, edge_index_train)
            link_loss = criterion(link_predictions.squeeze(), labels_train)
            contrastive_loss = link_predictor.contrastive_loss(com1, com2)
            loss = link_loss  + t * contrastive_loss
            
            # backward
            loss.backward()
            optimizer.step()
        
            # validation
            link_predictor.eval()
            with torch.no_grad():
                val_predictions, _, _, _ , _, _, _ = link_predictor(data_real, snp_gene_real, node_feature, adj_real, data_str, str_inter, feature_str, adj_feature, edge_index_val)
                val_predictions = val_predictions.squeeze().cpu().numpy()
                labels_val_np = labels_val.cpu().numpy()
                val_loss = criterion(torch.tensor(val_predictions, dtype=torch.float, device=device), labels_val).item()
            
            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early stopping at epoch {epoch} due to no improvement in validation loss for {patience} epochs.")
                    break 
                
            # AUC、AUPR和F1, ACC, 
            auc = roc_auc_score(labels_val_np, val_predictions)
            aupr = average_precision_score(labels_val_np, val_predictions)
            f1 = f1_score(labels_val_np, (val_predictions > 0.5).astype(int))
            acc = accuracy_score(labels_val_np, (val_predictions > 0.5).astype(int))
            
            print(f"Fold {fold + 1}, epoch: {epoch}, AUC: {auc}, AUPR: {aupr}, F1: {f1}, loss: {val_loss}")
            
        all_fold_auc.append(auc)
        all_fold_aupr.append(aupr)
        all_fold_f1.append(f1)
        all_fold_acc.append(acc)

        link_predictor.eval()
        with torch.no_grad():
            test_probabilities,emd, _, _, _, _, att = link_predictor(data_real, snp_gene_real, node_feature, adj_real, data_str, str_inter, feature_str, adj_feature, test_edge_index)
            test_probabilities = test_probabilities.squeeze().cpu().numpy()  # 取出预测的链接概率
            test_edge_pre = np.hstack([test_edge, test_probabilities.reshape(-1, 1)])
            test_edge_pre = pd.DataFrame(test_edge_pre, columns=['SNP', 'gene_name', 'pred'])
            test_edge_pre = test_edge_pre.groupby(['SNP']).apply(lambda x: x.sort_values(by='pred', ascending=False)).reset_index(drop=True)
            test_edge_pre['rank'] = test_edge_pre.groupby(['SNP']).cumcount() + 1
            current_file = os.path.dirname(os.path.abspath(__file__))
            file_name = current_file + f"/Result/prediction_{fold}_{args.dataset}.txt"
            np.savetxt(file_name, test_edge_pre, fmt='%s') 

            # save att
            # beta_cpu = att.detach().cpu().numpy()
            # emdd_cpu = emd.detach().cpu().numpy()
            # beta_reshaped = beta_cpu.squeeze(-1).T
            # np.save(f'{args.dataset}/beta_values_{fold}.npy', beta_reshaped)
            # np.savetxt(f'{args.dataset}/emd_values.txt', emdd_cpu, fmt='%d')
    
    # AUC、AUPR and F1
    average_auc = np.mean(all_fold_auc)
    average_aupr = np.mean(all_fold_aupr)
    average_f1 = np.mean(all_fold_f1)
    average_acc = np.mean(all_fold_acc)

    # compute std
    std_auc = np.std(all_fold_auc)
    std_aupr = np.std(all_fold_aupr)
    std_f1 = np.std(all_fold_f1)
    std_acc = np.std(all_fold_acc)


    # save result
    average_auc = round(average_auc, 3)
    average_aupr = round(average_aupr, 3)
    average_f1 = round(average_f1, 3)
    average_acc = round(average_acc, 3)

    std_auc = round(std_auc, 3)
    std_aupr = round(std_aupr, 3)
    std_f1 = round(std_f1, 3)
    std_acc = round(std_acc, 3)

    # print result
    print(f"\n10-Fold Cross-Validation Average AUC: {average_auc} ± {std_auc}")
    print(f"10-Fold Cross-Validation Average AUPR: {average_aupr} ± {std_aupr}")
    print(f"10-Fold Cross-Validation Average F1: {average_f1} ± {std_f1}")
    print(f"10-Fold Cross-Validation Average ACC: {average_acc} ± {std_acc}")
    
    # clear memory
    del dual_graph_transformer
    del link_predictor
    torch.cuda.empty_cache()

