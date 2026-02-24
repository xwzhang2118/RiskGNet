# RiskGNet

## 1. Project overview
RiskGNet is a graph neural network framework for predicting associations between single nucleotide polymorphisms (SNPs) and genes. It builds and trains a dual-graph model that leverages a real interaction graph and a feature-based (sequence/similarity) graph, and uses contrastive learning together with link prediction. The implementation uses PyTorch and PyTorch Geometric (PyG).

## 2. Repository structure (key files)
- `RiskGNet.py`: Main script containing model definitions and training loop with cross-validation (includes model classes, training, and evaluation logic).
- `Evaluation.py`: Script to compute ranking-based evaluation metrics (Precision@K, Recall@K, MAP@K, NDCG@K) from prediction outputs in `Result/`.
- `STAD/`: Example dataset folder containing files required by the scripts:
  - `adj_real_2.txt`: Real adjacency matrix (upper-triangular or symmetric format).
  - `data_2.pth`: PyTorch-saved graph object (e.g., `HeteroData`) used for the real graph.
  - `feature_str.txt`: Node features used for building a k-NN similarity graph (e.g., sequence features).
  - `node_feature.txt`: Node feature vectors for all nodes (SNPs first, then genes).
  - `snp_list_STAD.txt`: List of test SNPs.
  - `snp_list.txt`: Full list of SNPs (used for indexing/splitting).
  - `STAD_l2g.txt`: Reference SNP-to-gene associations (used for evaluation).
  - `test_edge_index.txt` / `test_edge_name.txt`: Index and name files for test edges.
- `Result/`: Output directory for prediction files. Example files: `prediction_0_STAD.txt` ... `prediction_4_STAD.txt` (one per CV fold/run).

## 3. Requirements & dependencies
Recommended Python: 3.8+.

Primary dependencies (high level):
- torch (PyTorch)
- torch-geometric (PyG) and its supplementary packages (`torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`, etc.)
- scikit-learn
- pandas
- numpy

Note: PyTorch Geometric must be installed for a PyTorch/CUDA combination that matches your environment—use the official PyG installation instructions to pick correct wheels.

Example installation commands (adjust for your CUDA / PyTorch version):

```bash
# Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# Base Python packages
pip install numpy pandas scikit-learn
# Install PyTorch (replace with the proper CUDA version if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Install PyG (follow the PyG docs for the correct wheel for your setup)
pip install torch-geometric
```

If `pip install torch-geometric` fails, follow the platform-specific installation instructions at https://pytorch-geometric.readthedocs.io/.

## 4. Quick start: training
Run training from the repository root (where `RiskGNet.py` and `STAD/` are located):

```bash
python RiskGNet.py --dataset STAD --epochs 200 --in_dim 128 --hidden_dim 256 --out_dim 128 --dropout 0.4 --num_heads 4 --num_layers 4 --learning_rate 0.007 --temperature 0.1 --patience 10 --t 0.1 --k 3
```

Notes on arguments:
- `--dataset`: name of the dataset folder (default: `STAD`).
- Other flags correspond to `args_parser()` inside `RiskGNet.py` and can be tuned as needed.

After training, the script will produce prediction files in `Result/` (for example `prediction_0_STAD.txt`, etc.). These files are used by the evaluation script.

## 5. Evaluation
After predictions are produced in `Result/`, run:

```bash
python Evaluation.py
```

`Evaluation.py` will read `Result/prediction_{0..4}_STAD.txt` files and compute the mean and standard deviation of Precision@5, Recall@5, MAP@5, and NDCG@5 across the prediction files.

## 6. File and format conventions
- Prediction files (`Result/prediction_*.txt`) are space-separated with columns: `region gene post_prob rank`. `Evaluation.py` treats the first column as the region/SNP identifier.
- `STAD/STAD_l2g.txt` should contain columns `rsId` and `gene_symbol`; `Evaluation.py` renames these to `snp` and `gene` respectively.

## 7. Troubleshooting
- CUDA not found: install a PyTorch build that includes CUDA matching your system, or run on CPU.
- PyG installation issues: use the PyG installation helper and choose the correct wheel for your PyTorch and CUDA versions.
- File path or parsing issues: `RiskGNet.py` constructs dataset paths relative to its file location. Ensure `STAD/` exists and files use expected delimiters/encodings.
- Shape/type mismatch errors: verify `node_feature.txt` and `feature_str.txt` have the expected shapes and that SNP/gene indexing matches assumptions in `RiskGNet.py`.

## 8. License & contact
No license file is included in this repository. Please confirm licensing with the project owner before reuse or redistribution.