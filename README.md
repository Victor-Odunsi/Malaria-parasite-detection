# Malaria-parasite-detection

# 🧬 Malaria Parasite Detection using YOLOv11n

[![YOLOv11](https://img.shields.io/badge/Model-YOLOv11n-blue)](https://github.com/ultralytics/ultralytics)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![Deploy](https://img.shields.io/badge/Deploy-HuggingFace-yellow)](https://huggingface.co/spaces)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

---

### 🧠 Project Overview

This project leverages **YOLOv11n**, a lightweight and efficient object detection model, to identify **infected** and **uninfected** red blood cells from microscopic blood smear images.  
It aims to **automate malaria screening**, reducing diagnostic time and the burden on laboratory professionals.

> 🩸 _"Transforming microscopic malaria detection through AI-driven precision and accessibility."_

---

## 🧩 Architecture Overview

![System Architecture](./Malaria%20Parasite%20Detection%20Architecture%20-%20YOLOv11n.png)

The system consists of four key layers:

1. **Training Layer** – Preprocessing, fine-tuning, and exporting trained weights.  
2. **CI/CD Pipeline** – Automates model testing and deployment to Hugging Face.  
3. **Deployment Layer** – Serves model predictions via FastAPI and Streamlit.  
4. **Monitoring Layer** – Tracks performance metrics via Prometheus & Grafana.

---

## 📚 Dataset

| Source | Description | Format |
|:--|:--|:--|
| [NIH Malaria Dataset](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html) | Real microscopic HPF images | JPEG |
| [Kaggle Malaria Cell Images](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) | Single-cell labeled dataset | PNG |

**Dataset Summary:**
- 🧾 **Train:** 943 images  
- 🧪 **Validation:** 236 images  
- 🧬 **Classes:** `infected`, `uninfected`

---

## ⚙️ Training Configuration

```yaml
# data.yaml
train: data/malaria_cells/images/train
val: data/malaria_cells/images/val

nc: 2
names: ['infected', 'uninfected']
