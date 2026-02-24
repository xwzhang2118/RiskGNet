import os
import numpy as np
import pandas as pd

# =========================
# 1. defining evaluation metrics
# =========================
def precision_at_k(pred_genes, true_genes, k):
    hits = [gene in true_genes for gene in pred_genes[:k]]
    return np.sum(hits) / k

def recall_at_k(pred_genes, true_genes, k):
    hits = [gene in true_genes for gene in pred_genes[:k]]
    return np.sum(hits) / len(true_genes) if true_genes else 0

def map_at_k(pred_genes, true_genes, k):
    ap_sum = 0
    correct_predictions = 0
    for idx, gene in enumerate(pred_genes[:k], start=1):
        if gene in true_genes:
            correct_predictions += 1
            ap_sum += correct_predictions / idx
    return ap_sum / len(true_genes) if true_genes else 0

def ndcg_at_k(pred_genes, true_genes, k):
    dcg = 0
    idcg = 0
    for idx, gene in enumerate(pred_genes[:k]):
        if gene in true_genes:
            dcg += 1 / np.log2(idx + 2)  # idx+2 修正 DCG
    for idx in range(min(len(true_genes), k)):
        idcg += 1 / np.log2(idx + 2)
    return dcg / idcg if idcg > 0 else 0

# =========================
# 2. computing metrics for each snp
# =========================
def calculate_metrics_at_k(prediction, reference, k=5):
    results = []
    ref_dict = reference.groupby('snp')['gene'].apply(set).to_dict()
    
    for region, group in prediction.groupby('region'):
        group_sorted = group.sort_values('rank').head(k)
        pred_genes = group_sorted['gene'].tolist()
        true_genes = ref_dict.get(region, set())
        if not true_genes:
            continue
        
        precision = precision_at_k(pred_genes, true_genes, k)
        recall = recall_at_k(pred_genes, true_genes, k)
        map_score = map_at_k(pred_genes, true_genes, k)
        ndcg = ndcg_at_k(pred_genes, true_genes, k)
        
        results.append({
            'region': region,
            'Precision@K': precision,
            'Recall@K': recall,
            'MAP@K': map_score,
            'NDCG@K': ndcg
        })
    
    return results

# =========================
# 3. main function
# =========================
if __name__ == '__main__':
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    reference = pd.read_csv(current_file_path + '/STAD/STAD_l2g.txt', sep='\t')
    reference = reference[['rsId','gene_symbol']]
    reference.columns = ['snp','gene']
    
    test_snp = pd.read_csv(current_file_path + '/STAD/snp_list_STAD.txt', sep='\t')
    test_set = set(test_snp['SNP'])

    precision_all = []
    recall_all = []
    mrr_all = []
    map_all = []
    ndcg_all = []

    for i in range(0, 5):
        
        filename = current_file_path + f'/Result/prediction_{i}_STAD.txt'
        prediction = pd.read_csv(filename, sep=' ')
        prediction.columns = ['region', 'gene','post_prob','rank']
        prediction = prediction[prediction['region'].isin(test_set)]

        # compute metrics
        metrics = calculate_metrics_at_k(prediction, reference, k=5)
        metrics_df = pd.DataFrame(metrics)

        # average metrics
        avg_metrics = metrics_df[['Precision@K','Recall@K','MAP@K','NDCG@K']].mean()
        precision_all.append(avg_metrics['Precision@K'])
        recall_all.append(avg_metrics['Recall@K'])
        map_all.append(avg_metrics['MAP@K'])
        ndcg_all.append(avg_metrics['NDCG@K'])

    # output results
    print(f'Precision: {np.mean(precision_all):.3f} ± {np.std(precision_all):.3f}')
    print(f'Recall: {np.mean(recall_all):.3f} ± {np.std(recall_all):.3f}')
    print(f'MAP: {np.mean(map_all):.3f} ± {np.std(map_all):.3f}')
    print(f'NDCG: {np.mean(ndcg_all):.3f} ± {np.std(ndcg_all):.3f}')
