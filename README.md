# CARMA: Enhanced Compositionality in LLMs via Advanced Regularisation and Mutual Information Alignment
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository contains the code for the paper "CARMA: Enhanced Compositionality in LLMs via Advanced Regularisation and Mutual Information Alignment" (EMNLP 2025).

## Overview

CARMA (**C**ompositionality in LLMs via **A**dvanced **R**egularisation and **M**utual Information **A**lignment) is a framework designed to improve compositional generalisation in large language models. It combines mutual information regularisation with layer-wise stability regularisation to mitigate information fragmentation and preserve robust compositional representations.

## Features

- **Mutual Information Regularisation:** Preserves token dependencies across layers to enhance structured compositionality.
- **Layer-Wise Stability Regularisation:** Reduces representational drift across consecutive layers and improves model stability.
- **Flexible Model Integration:** Applies to pre-trained language models with minimal modifications.
- **Comprehensive Evaluation:** Supports inverse dictionary modelling and sentiment compositionality tasks.
- **Robustness Interventions:** Includes synonym replacement and Constituent-Aware Pooling experiments.

## Setup

Use the provided `environment.yml` or install the dependencies with pip:

```bash
conda env create -f environment.yml
conda activate CARMA
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
```

Alternatively:

```bash
git clone https://github.com/nura-j/CARMA.git
cd CARMA
pip install -r requirements.txt
```

## Usage

The main entry point is `src/main.py`. Training the model with CARMA's mutual information and stability regularisation:

```bash
python -m src.main \
  --model_name GPT2 \
  --dataset_name idm \
  --data_path "./data/wordnet_data_definitions.json" \
  --carma_weight 0.3 \
  --mi_weight 0.5 \
  --stability_weight 0.2 \
  --mi_end_layer 6 \
  --stability_end_layer 4 \
  --train \
  --save_model \
  --test \
  --save_test_results
```

For sentiment compositionality tasks:

```bash
python -m src.main \
  --model_name GPT2 \
  --dataset_name sst \
  --data_path "stanfordnlp/sst" \
  --carma_weight 0.3 \
  --mi_weight 0.5 \
  --stability_weight 0.2 \
  --mi_end_layer 6 \
  --stability_end_layer 4 \
  --train \
  --save_model \
  --test \
  --save_test_results
```

### Experiment arguments

| Argument | Purpose |
| --- | --- |
| `--model_name` | Model to train or evaluate; `GPT2` is the default. |
| `--dataset_name` | `idm` or `sst`. |
| `--data_path` | Path to the processed IDM data or Hugging Face dataset name. |
| `--model_path` | Path to a saved fine-tuned model. |
| `--carma_weight` | Weight of the combined CARMA objective. |
| `--mi_weight` | Weight of mutual information regularisation. |
| `--stability_weight` | Weight of layer-wise stability regularisation. |
| `--mi_start_layer` / `--mi_end_layer` | Layers covered by mutual information regularisation. |
| `--stability_start_layer` / `--stability_end_layer` | Layers covered by stability regularisation. |
| `--train` / `--test` | Run training and/or evaluation. |
| `--save_model` / `--save_test_results` | Save model checkpoints and test outputs. |

### Robustness experiments

Evaluate synonym replacement:

```bash
python -m src.main \
  --intervention \
  --intervention_type synonym \
  --intervention_percentage 0.1 \
  --save_test_results \
  --model_name GPT2 \
  --model_path "path/to/saved/model" \
  --correct_pred_path "path/to/correct_predictions" \
  --dataset_name idm \
  --data_path "./data/wordnet_data_definitions.json"
```

Evaluate Constituent-Aware Pooling:

```bash
python -m src.main \
  --intervention \
  --intervention_type cap \
  --CAP_start_layer 1 \
  --grouping_protocol max \
  --intervention_percentage 0.1 \
  --save_test_results \
  --model_name GPT2 \
  --model_path "path/to/saved/model" \
  --correct_pred_path "path/to/correct_predictions" \
  --dataset_name idm \
  --data_path "./data/wordnet_data_definitions.json"
```

## Outputs

Training history, plots, model checkpoints, and test results are written according to the save flags and paths supplied in the command. Generated results and visualisations are stored locally and should not be committed unless required for a specific experiment.

## Repository layout

```text
.
├── README.md
├── environment.yml
├── requirements.txt
├── LICENSE
├── data/
│   └── wordnet_data_definitions.json
└── src/
    ├── main.py                  # Command-line entry point
    ├── training.py              # Fine-tuning and CARMA optimisation
    ├── optm_losses.py           # Regularisation objectives
    ├── testing.py               # Model evaluation
    ├── CAP.py                   # Constituent-Aware Pooling intervention
    ├── interventions.py
    ├── dataset.py               # Dataset loading and processing
    ├── model_setup.py           # Transformers and TransformerLens setup
    ├── synonym_generator.py
    ├── load_saved_predictions.py
    └── utils.py
```

## Results

- CARMA improves compositional generalisation compared with fine-tuned baselines.
- Layer-wise analysis shows the effect of CARMA on compositional structure preservation.
- Synonym replacement and CAP experiments confirm increased robustness to lexical perturbations.

Full experimental results are available in the [EMNLP 2025 proceedings](https://aclanthology.org/2025.emnlp-main.822/).

## Citation

If you use CARMA in your research, please cite:

```bibtex
@inproceedings{aljaafari-etal-2025-carma,
    title = "{CARMA}: Enhanced Compositionality in {LLM}s via Advanced Regularisation and Mutual Information Alignment",
    author = "Aljaafari, Nura  and
      Carvalho, Danilo  and
      Freitas, Andre",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.822/",
    doi = "10.18653/v1/2025.emnlp-main.822",
    pages = "16239--16259",
    ISBN = "979-8-89176-332-6",
    abstract = "Large language models (LLMs) struggle with compositional generalisation, limiting their ability to systematically combine learned components to interpret novel inputs. While architectural modifications, fine-tuning, and data augmentation improve compositionality, they often have limited adaptability, face scalability constraints, or yield diminishing returns on real data. To address this, we propose CARMA, an intervention that enhances the stability and robustness of compositional reasoning in LLMs while preserving fine-tuned performance. CARMA employs mutual information regularisation and layer-wise stability constraints to mitigate feature fragmentation, ensuring structured representations persist across and within layers. We evaluate CARMA on inverse dictionary modelling and sentiment classification, measuring its impact on semantic consistency, performance stability, and robustness to lexical perturbations. Results show that CARMA reduces the variability introduced by fine-tuning, stabilises token representations, and improves compositional reasoning. While its effectiveness varies across architectures, CARMA{'}s key strength lies in reinforcing learned structures rather than introducing new capabilities, making it a scalable auxiliary method. These findings suggest that integrating CARMA with fine-tuning can improve compositional generalisation while maintaining task-specific performance in LLMs."
}
```

## License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues regarding this code, or for paper-related inquiries:
- Open an issue in this repository
- Contact: nuraaljaafari@gmail.com
