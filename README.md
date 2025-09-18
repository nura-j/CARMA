
# CARMA: Enhanced Compositionality in LLMs via Advanced Regularisation and Mutual Information Alignment

## Overview  
CARMA (**C**ompositionality in LLMs via **A**dvanced **R**egularisation and **M**utual Information **A**lignment) is a framework designed to improve **compositional generalisation (CG)** in large language models (LLMs). CARMA enhances structured compositionality through **Mutual Information Regularisation** and **Layer-Wise Stability Regularisation**, mitigating information fragmentation and ensuring robust compositional representations.

## Features  
- **Mutual Information Regularisation:** Preserves token dependencies across layers to enhance structured compositionality.  
- **Layer-Wise Stability Regularisation:** Reduces representational drift across consecutive layers, improving model stability.  
- **Flexible Model Integration:** CARMA can be applied to pre-trained LLMs with minimal modifications.  
- **Comprehensive Evaluation:** Benchmarked on **Inverse Dictionary Modelling (IDM)** and **Sentiment Compositionality (SC)** tasks.  

## Repository Structure  
```
📂 CARMA  
 ├── 📁 src/                  # Source code for model training and evaluation  
 ├── 📁 data/                 # Processed datasets (WordNet)  
 ├── 📁 plots/              # Visualizations and result plots  
 ├── 📁 results/              # Generated model outputs and performance logs  
 ├── 📄 requirements.txt      # Dependencies for running CARMA  
 ├── 📄 README.md             # Project documentation  
 ├── 📄 LICENSE               # License information  
 ├── 📄 environment.yaml           # Configuration file for experiments  
```

## Installation  
To set up CARMA, first **clone the repository** and install dependencies:  

```bash
git clone https://github.com/your-repo/CARMA.git
cd CARMA
pip install -r requirements.txt
```

## Usage  


### **1. Training the Model with CARMA**  
To train a model with CARMA’s **Mutual Information and Stability Regularisation**, use:  

```bash
python -m src.main --model_name GPT2 --dataset_name idm --data_path "./data/wordnet_data_definitions.json" --carma_weight 0.3 --mi_weight 0.5 --stability_weight 0.2 --mi_end_layer 6 --stability_end_layer 4 --train --save_model --test --save_test_results 
```

For **Sentiment Compositionality (SC)** tasks:  

```bash
python -m src.main --model_name GPT2 --dataset_name sst --data_path "stanfordnlp/sst" --carma_weight 0.3 --mi_weight 0.5 --stability_weight 0.2 --mi_end_layer 6 --stability_end_layer 4 --train --save_model --test --save_test_results  

```

### **2. Running Experiments**  
Evaluate performance under **Synonym Replacement** and **Constituent-Aware Pooling (CAP)**:  

```bash
python src.main --intervention --intervention_type synonym --intervention_percentage 0.1 --save_test_results --model_name GPT2 --model_path "path/to/saved/model" --correct_pred_path "path/correct_results" --dataset_name idm --data_path "./data/wordnet_data_definitions.json" 
```
```bash
python -m src.main --intervention --intervention_type CAP --CAP_start_layer 1 --grouping_protocol max --intervention_percentage 0.1 --save_test_results --model_name GPT2 --model_path "path/to/saved/model" --correct_pred_path "path/correct_results" --dataset_name idm --data_path "./data/wordnet_data_definitions.json"
```

## Results  
- CARMA significantly **improves CG** compared to fine-tuned baselines.  
- **Layer-wise analysis** reveals CARMA’s impact on compositional structure preservation.  
- **Synonym Replacement and CAP experiments** confirm increased robustness to lexical variations.  

Full experimental results are available in **PAPER_LINK**.  

## Citation  
If you use CARMA in your research, please cite:  
```bibtex
@article{aljaafari2025carma,
  title={Carma: Enhanced compositionality in llms via advanced regularisation and mutual information alignment},
  author={Aljaafari, Nura and Carvalho, Danilo S and Freitas, Andr{\'e}},
  journal={arXiv preprint arXiv:2502.11066},
  year={2025}
}
```

## License  
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact  
For questions or collaboration, please reach out to **nuraaljaafari@gmail.com**.
