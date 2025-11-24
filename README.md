<img width="700" height="160" alt="fig_flowchart_small" src="https://github.com/user-attachments/assets/cbd2a000-49f8-49eb-a839-deaa8e64444e" />

**HiGEC**  
**Hierarchy Generation and Extended Classification Framework**  

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)  
[![OpenML](https://img.shields.io/badge/OpenML-datasets-orange)](https://www.openml.org)  

HiGEC is a Python framework for enhancing multi-class classification through **automated hierarchy generation (HG)** and **flexible hierarchy exploitation (HE)** strategies. It supports hybrid approaches that integrate hierarchical and flat classifier outputs.

---
<details>
<summary>🔧 Installation</summary>

```bash
git clone https://github.com/alagoz/higec.git
cd higec
pip install -r requirements.txt
```

**Dependencies:**  
`numpy` `scipy` `matplotlib` `scikit-learn` `scikit-learn-extra` `proglearn` `xgboost` `lightgbm`

---
</details>

<details> <summary>⚡ Key Features</summary>
  
� **Automatic hierarchy generation** from flat class labels
  
🧩 **Hybrid HE+F classification strategies**
  
🖇️ Support for **any scikit-learn compatible classifier**
  
📊 **Benchmark-ready** with OpenML integration
  
🌳 **Visualization tools** for hierarchy inspection

---
</details>

<details> <summary>🚀 Quick Start</summary>

Run the example:
```bash
python run_higec_example.py
```

Pipeline:
1. Downloads OpenML dataset

2. Trains flat classifier baseline

3. Generates class hierarchy

4. Evaluates hierarchical approach

---
</details>

<details> <summary>🛠 Core Components</summary>

| File       | Purpose                           |
|------------|-----------------------------------|
| `HG.py`    | Hierarchy generation              |
| `HE.py`    | Hierarchy exploitation            |
| `hdc.py`   | Divisive clustering               |
| `utils.py` | Data handling & visualization     |

---
</details>

<details> <summary>🧪 Customization</summary>

Adjust parameters in 'run_higec_example.py':

```bash
DID = 46264                       # OpenML dataset ID
HiGEC = 'CCM[HAC|COMPLETE]-LCPN[ETC]+F[XGB]'  # HG + HE scheme
CLF_NAME_FC = 'RF'                # Flat classifier
```

Available classifiers: `RF`, `XGB`, `ETC`, `LGB`.

---
</details>

<details> <summary>📈 Example Output</summary>

```bash
Extended Linkage Table:

node_id:0, node_type:parent, subsets:[[0], [1,2,3,4]], branch_ids:[0,7], parent_id:None
node_id:1, node_type:parent, subsets:[[3,4],[1,2]], branch_ids:[5,6], parent_id:0
```

```bash
Performance Comparison:

- Flat Classification (RF) (f1): 0.3517 in 0.4309 seconds
- HiGEC: CCM[HAC|COMPLETE]-LCPN[ETC]+F[XGB] (f1): 0.3700 in 1.1853 seconds
```

Generated Hierarchy:  
![example_hierarchy](https://github.com/user-attachments/assets/96e78795-541b-41a1-a7bb-a945b65411fa)

---
</details>

<details>
<summary>📊 Benchmark Results</summary>

HiGEC was evaluated on **100 multi-class tabular datasets**, showing consistent F1-score gains over flat classification (FC), particularly with hybrid HE+F configurations.

---

### Mean F1 Comparison (HiGEC vs FC)

<img width="1476" height="387" alt="fig_mcm_higec_vs_fc" src="https://github.com/user-attachments/assets/614581db-e193-44dc-a5d2-998db14887b5" />


### Mean F1 Scores & Standard Deviations

![table](https://github.com/user-attachments/assets/b7b5a00c-e597-4576-bc19-b1d881d66541)

---

**Download raw results (F1 scores per dataset):**  
- [f1_scores_fc_vs_higec.csv](./results/f1_scores_fc_vs_higec.csv) – Contains per-dataset F1-scores of FC and selected 9 HiGEC algorithms.  
- Columns: `index`, `short`, `RF`, `XGB`, `ETC`, `LGB`, `LCN[XGB]+`, `LCPN[ETC]+F[XGB]`, `LCPN[RF]+F[XGB]`, `LCPN[XGB]+F[RF]`, `LCL[XGB]+F[RF]`, `LCPN[RF]+F[RF]`, `LCL[RF]+F[XGB]`, `LCPN[LGB]+F[XGB]`, `LCPN[XGB]+F[XGB]` 

**Download mean performance metrics for all FC algorithms:**  
- [fc_mean_performance.csv](./results/fc_mean_performance.csv) – Contains mean scores across datasets for each FC algorithm.  
- Columns: `index`, `short`, `mean_f1_xgb`, `mean_f1_catb`, ... , `mean_acc_xgb`, `mean_acc_catb`, ... , `mean_auc_xgb`, `mean_auc_catb`, ... , `total_dur_xgb`, `total_dur_catb`, ...


These CSV files allow full reproducibility and further statistical analysis of HiGEC’s performance compared to FC.

---
</details>

<details> <summary>📖 References</summary>

For more details on methodology, datasets, and evaluations, see the HiGEC GitHub repository.

</details>
