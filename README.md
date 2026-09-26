# Context-Aware AutoML for Sparse SAE Intervention

This repository contains the implementation for our BIBM 2026 workshop paper.

We formulate clinical fairness intervention as a multi-objective AutoML problem over sparse autoencoder features, distinguishing counterfactual cases where responses should remain consistent from those where clinically meaningful differences should be preserved.


### File Description
Below is the detailed organization of the core components:
```
.
├── 📖README.md                        # Project documentation
├── LICENSE
├── ⚙️requirements.txt                 # Project configuration and dependencies
├── .gitignore
|
├──📂configs
|   └── ⚙️default.yaml                 # Configuration file
|
├──📂data
│   ├── 📖README.md               
│   └── 📂processed
|        └── .gitkeep
│      
├──📂src
│   ├── 📜model.py              
│   ├── 📜prompt.py     
│   ├── 📜sae_utils.py
|   ├── 📜intervention.py
|   ├── 📜metrics.py
|   └── 📜data_utils.py
|
├──📂scripts
│   ├── 📜prepare_equitymedqa.py              
│   ├── 📜discover_sae_features.py     
│   ├── 📜causal_filter.py
|   ├── 📜run_automl.py
|   ├── 📜evaluate_medqa.py
|   ├── 📜evaluate_ccmanual.py
|   └── 📜activation_shift_analysis.py
|
└──📂results
    ├── 📊main_results.csv             
    ├── 📊medqa_results.csv    
    └── 📊external_robustness.csv

```



 
