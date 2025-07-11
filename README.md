# Medical Vision-Language Model Optimization Framework

This framework provides a comprehensive evaluation and optimization system for medical vision-language models using DSPy. It includes evaluation metrics, multiple medical datasets, and optimization strategies.

## Features

- **5 Medical VLM Experiments**:
  - VQA RAD: Visual Question Answering on Radiology images
  - CheXpert: Chest X-ray classification
  - DDI Disease: Dermatology disease diagnosis
  - DDI Skintone: Skin tone classification
  - Gastrovision: Gastroenterology endoscopy classification

- **40+ Evaluation Metrics**: Including exact match, F1 score, BLEU, ROUGE, CIDEr, WER, and more

- **3 Optimization Strategies**:
  - BootstrapFewShotWithRandomSearch
  - MIPROv2
  - SIMBA

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running Single Experiments

```bash
python scripts/run_experiment.py \
  --experiment vqa_rad \
  --model "your-model-name" \
  --api_base "your-api-base" \
  --api_key "your-api-key"
```

### Running Batch Experiments

```bash
python scripts/batch_run.py \
  --model "your-model-name" \
  --api_base "your-api-base" \
  --api_key "your-api-key" \
  --experiments vqa_rad chexpert
```

### Available Experiments

- `vqa_rad`: Visual Question Answering on Radiology images
- `chexpert`: Chest X-ray classification
- `ddi_disease`: Dermatology disease diagnosis
- `ddi_skintone`: Skin tone classification
- `gastrovision`: Gastroenterology endoscopy classification

## Configuration

The framework uses configurable paths in `config/paths.py`. Update the `BASE_DATA_DIR` to point to your data directory:

```python
BASE_DATA_DIR = Path("/your/data/directory")
```

## Directory Structure

```
medvlm_optimization/
├── config/               # Configuration files
├── src/
│   ├── experiments/      # Individual experiment implementations
│   ├── utils/           # Utility functions
│   ├── metrics.py       # Evaluation metrics
│   └── main.py          # Main execution logic
├── scripts/             # CLI scripts
├── outputs/             # Generated logs and results
└── requirements.txt     # Python dependencies
```

## Output

Results are logged to `outputs/logs/` with detailed experiment information and performance metrics.

## Data Requirements

The framework expects data in the following structure:
- CheXpert: CSV files with image paths and labels
- DDI: CSV metadata files and image directories
- Gastrovision: CSV files with base64-encoded images
- VQA RAD: Loads from HuggingFace datasets