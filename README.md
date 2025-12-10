# MVP: Multi-View Prediction for Stable GUI Grounding

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 🎯 Overview

MVP (Multi-View Prediction) is a training-free framework that addresses the critical issue of **coordinate prediction instability** in GUI grounding models. Our method significantly improves grounding performance by aggregating predictions from multiple carefully crafted views, effectively distinguishing stable coordinates from outliers.

![MVP Framework](assets/framework.png)

*Figure: The MVP framework consists of two main components: (1) Attention-Guided View Proposal that generates diverse cropped views based on instruction-to-image attention, and (2) Multi-Coordinates Clustering that ensembles predictions by selecting the centroid of the densest spatial cluster.*

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ZJUSCL/MVP.git
cd MVP
pip install -r requirements.txt
```

### Datasets Download from Hugging Face

```bash
# Install huggingface_hub for dataset download
pip install huggingface_hub

# Download UI-Vision dataset
huggingface-cli download ServiceNow/ui-vision --local-dir ./data/ui-vision

# Download ScreenSpot-Pro dataset
huggingface-cli download likaixin/ScreenSpot-Pro --local-dir ./data/screenspot-pro

# Download OSWorld-G dataset
huggingface-cli download MMInstruction/OSWorld-G --local-dir ./data/osworld-g
```

### Models Download from Hugging Face

```bash
# Download UI-TARS-1.5-7B model
huggingface-cli download ByteDance-Seed/UI-TARS-1.5-7B --local-dir ./models/UI-TARS-1.5-7B

# Download GTA1-7B model
huggingface-cli download HelloKKMe/GTA1-7B --local-dir ./models/GTA1-7B

# Download Qwen3VL-8B model
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir ./models/Qwen3-VL-8B-Instruct

# Download Qwen3VL-32B model
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct --local-dir ./models/Qwen3-VL-32B-Instruct
```

### Alternative: Using Git LFS

```bash
# For large models, you can also use Git LFS
git lfs install
git clone https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B ./models/UI-TARS-1.5-7B
git clone https://huggingface.co/HelloKKMe/GTA1-7B ./models/GTA1-7B
git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct ./models/Qwen3-VL-8B-Instruct
git clone https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct ./models/Qwen3-VL-32B-Instruct
```

## 📊 Performance

### ScreenSpot-Pro Benchmark Results

| Model | Development | Creative | CAD | Scientific | Office | OS | Overall |
|-------|-------------|----------|-----|------------|--------|----|---------|
| **UI-TARS-1.5-7B** | 36.4 | 38.1 | 20.5 | 49.6 | 68.7 | 31.5 | 41.9 |
| **+ MVP** | 51.8↑15.4 | 50.0↑11.9 | 53.3↑32.8 | 57.9↑8.3 | 73.0↑4.3 | 54.6↑23.1 | 56.1↑14.2 |
| **GTA1-7B** | 43.4 | 44.8 | 44.4 | 55.9 | 74.8 | 35.2 | 49.8 |
| **+ MVP** | 58.9↑15.5 | 52.6↑7.8 | 60.2↑15.8 | 63.0↑7.1 | 79.1↑4.3 | 56.1↑20.9 | 61.7↑11.9 |
| **Qwen3VL-8B** | 52.8 | 49.1 | 49.0 | 56.7 | 75.2 | 50.5 | 55.0 |
| **+ MVP** | 61.5↑8.7 | 60.2↑11.1 | 61.3↑12.3 | 67.3↑10.6 | 82.6↑7.4 | 62.8↑12.3 | 65.3↑10.3 |
| **Qwen3VL-32B** | 43.1 | 54.4 | 57.5 | 62.6 | 73.0 | 42.3 | 55.3 |
| **+ MVP** | **71.6**↑28.5 | **69.3**↑14.9 | **74.7**↑17.2 | **70.5**↑7.9 | **87.4**↑14.4 | **73.5**↑31.2 | **74.0**↑18.7 |

## 🛠️ Evaluation Scripts

### Run All Experiments

We provide four main evaluation scripts for different model configurations:

```bash
# Run experiments for UI-TARS-1.5-7B and GTA1-7B
./eval_gta1.sh

# Run experiments for Qwen3VL-8B
./eval_qwen3vl8b.sh

# Run experiments for Qwen3VL-32B
./eval_uitars_1_5.sh

# Run all experiments sequentially
./eval_qwen3vl32b.sh
```

## 🔧 Core Components

### 1. Attention-Guided View Proposal
- Generates multiple cropped views based on instruction-to-image attention
- Focuses on relevant regions while maintaining context

### 2. Multi-Coordinates Clustering  
- Aggregates predictions from multiple views
- Uses density-based clustering to identify stable coordinates
- Selects centroid of densest cluster as final prediction

## 📄 Citation

If you find our work useful, please cite our paper:

```bibtex
@article{mvp2024,
  title={MVP: Multiple View Prediction Improves GUI Grounding},
  author={Yunzhu Zhang, Zeyu Pan, Shuheng Shen, Changhua Meng and Linchao Zhu},
  journal={arXiv preprint},
  year={2025},
  url={https://arxiv.org/abs/xxxx.xxxxx}
}
```

## 📜 License

This project is licensed under the Apache License 2.0.

## 📧 Contact

For questions about this work, please open an issue or contact [yunzhuzhang0918@gmail.com].