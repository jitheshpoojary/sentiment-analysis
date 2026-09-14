# 🛒 Amazon Reviews Sentiment Analysis

<p align="center">
  <strong>End-to-End NLP • Machine Learning • Deep Learning • Transformers • Deployment</strong>
</p>

<p align="center">
  <a href="https://sentiment-analysisgit-pnc7pgn9cvws5tkac2weto.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge" alt="Live Demo">
  </a>
  <a href="https://huggingface.co/jitheshpoojary/amazon-roberta-sentiment">
    <img src="https://img.shields.io/badge/🤗%20Model-Hugging%20Face-FFD21E?style=for-the-badge" alt="Hugging Face Model">
  </a>
  <a href="https://github.com/jitheshpoojary/sentiment-analysis">
    <img src="https://img.shields.io/badge/💻%20Source-GitHub-181717?style=for-the-badge" alt="GitHub">
  </a>
</p>

---

## 📌 Overview

This project focuses on **sentiment analysis of Amazon customer reviews** using Natural Language Processing (NLP), Machine Learning, Deep Learning, and Transformer-based models.

The objective is to automatically classify Amazon reviews into three sentiment categories:

| Label | Sentiment |
|---|---|
| 🔴 0 | Negative |
| 🟡 1 | Neutral |
| 🟢 2 | Positive |

The project follows a complete machine learning workflow — from **data understanding and exploratory analysis to feature engineering, model experimentation, Transformer fine-tuning, and deployment as an interactive web application**.

The final **RoBERTa-base** model achieved approximately:

> **91% Accuracy**  
> **80% Macro F1 Score**

---

## 🚀 Try the Live Application

### 👉 [Open Amazon Reviews Sentiment Analyzer](https://sentiment-analysisgit-pnc7pgn9cvws5tkac2weto.streamlit.app/)

Enter an Amazon review and the application predicts:

- 🎯 Sentiment
- 📊 Prediction confidence
- 📈 Probability distribution across all classes
- 📝 Review statistics
- 💡 Sentiment interpretation

---

## 🎯 Project Objectives

- Analyze sentiment from real-world **Amazon Reviews data**
- Build a complete NLP pipeline
- Compare traditional Machine Learning approaches with Deep Learning
- Explore Transformer-based sentiment classification
- Handle challenging Neutral sentiment classification
- Fine-tune RoBERTa for contextual text understanding
- Deploy the final model as an interactive Streamlit application

---

## 📊 Dataset

The project uses **Amazon Reviews data** for sentiment analysis, containing **600K+ customer reviews**.

The reviews represent real-world customer feedback and contain challenges commonly encountered in NLP applications, including:

- Informal language
- Spelling variations
- Short reviews
- Mixed sentiment
- Ambiguous opinions
- Class imbalance
- Context-dependent sentiment

### Sentiment Classes

```text
0 → Negative
1 → Neutral
2 → Positive
```

The original dataset and large generated artifacts are **not included in this GitHub repository** to keep the repository lightweight.

---

# 🔄 End-to-End Workflow

```text
                 Amazon Reviews Data
                         │
                         ▼
              ┌─────────────────────┐
              │ Data Understanding  │
              │       & EDA         │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Text Preprocessing  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Feature Engineering │
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Machine Learning       Deep Learning
              │                     │
              ▼                     ▼
       TF-IDF / BoW          LSTM / GRU
       Linear SVM            BiGRU + Attention
       LightGBM              FastText
              │                     │
              └──────────┬──────────┘
                         ▼
                  Transformer Models
                         │
                         ▼
                    RoBERTa-base
                         │
                         ▼
                 Hugging Face Hub
                         │
                         ▼
                Streamlit Web App
```

---

# 🧠 Models Explored

The project compares multiple approaches to understand how different text representations and architectures perform on sentiment classification.

## Machine Learning

- Bag of Words
- TF-IDF
- TF-IDF + Feature Selection
- TF-IDF + SVD
- Hybrid Feature Representations
- Linear SVM
- LightGBM

## Deep Learning

- LSTM
- GRU
- Weighted GRU
- BiGRU
- BiGRU with Attention
- FastText Embeddings

## Transformer Models

- DistilRoBERTa
- RoBERTa-base

---

# 🏆 Final Model — RoBERTa-base

The final model uses **RoBERTa-base**, a Transformer architecture capable of capturing contextual relationships between words.

### Model Configuration

| Parameter | Value |
|---|---|
| Architecture | RoBERTa-base |
| Task | 3-Class Sentiment Classification |
| Maximum Sequence Length | 256 |
| Batch Size | 8 |
| Learning Rate | `1e-5` |
| Training Epochs | 4 |
| Loss Function | Class-Weighted Cross Entropy |
| Classes | Negative / Neutral / Positive |

### Performance

| Metric | Score |
|---|---:|
| 🎯 Accuracy | **~91%** |
| 📊 Macro F1 | **~80%** |

The model performs strongly overall, while **Neutral sentiment remains the most challenging class** because neutral and mixed opinions can have overlapping linguistic patterns.

---

# 🤗 Hugging Face Model

The trained RoBERTa model is hosted separately on Hugging Face.

### 👉 [View the RoBERTa Sentiment Model](https://huggingface.co/jitheshpoojary/amazon-roberta-sentiment)

The Streamlit application loads the trained model directly from the Hugging Face Hub rather than storing the large model files inside the GitHub repository.

---

# 🌐 Interactive Streamlit Application

The project includes an interactive web application built with **Streamlit**.

### Application Features

```text
┌─────────────────────────────────────────┐
│       Amazon Sentiment Analyzer         │
├─────────────────────────────────────────┤
│                                         │
│  Enter your Amazon review               │
│                                         │
│  "The product quality is excellent..."  │
│                                         │
│            [ Analyze Review ]            │
│                                         │
├─────────────────────────────────────────┤
│                                         │
│  Prediction: 🟢 Positive                │
│  Confidence: XX%                        │
│                                         │
│  Negative   ███                         │
│  Neutral    ██                          │
│  Positive   █████████████               │
│                                         │
└─────────────────────────────────────────┘
```

### 🔗 Live Demo

**[🚀 Launch the Sentiment Analysis App](https://sentiment-analysisgit-pnc7pgn9cvws5tkac2weto.streamlit.app/)**

---

# 📓 Project Notebooks

The project is organized into six stages:

| Notebook | Description |
|---|---|
| `01_Data_Understanding_EDA.ipynb` | Data understanding and exploratory data analysis |
| `02_Text_Preprocessing.ipynb` | Text cleaning and preprocessing |
| `03_Feature_Engineering.ipynb` | Feature extraction and feature representation |
| `04_Machine_Learning.ipynb` | Classical ML models and evaluation |
| `05_Deep_Learning.ipynb` | LSTM, GRU, BiGRU and attention-based models |
| `06_Transformers_BERT.ipynb` | Transformer-based sentiment classification |

---

# 📁 Repository Structure

```text
sentiment-analysis/
│
├── 📓 01_Data_Understanding_EDA.ipynb
├── 📓 02_Text_Preprocessing.ipynb
├── 📓 03_Feature_Engineering.ipynb
├── 📓 04_Machine_Learning.ipynb
├── 📓 05_Deep_Learning.ipynb
├── 📓 06_Transformers_BERT.ipynb
│
├── 🐍 app.py
├── 📦 requirements.txt
├── 🚫 .gitignore
│
└── README.md
```

> **Note:** Large datasets, trained models, intermediate artifacts, and generated files are excluded from GitHub using `.gitignore`.

---

# 🛠️ Technology Stack

### Programming

- Python

### Data Science

- Pandas
- NumPy
- Scikit-learn

### NLP

- Text Preprocessing
- Bag of Words
- TF-IDF
- Feature Engineering
- FastText

### Machine Learning

- Linear SVM
- LightGBM

### Deep Learning

- PyTorch
- TensorFlow / Keras
- LSTM
- GRU
- BiGRU
- Attention Mechanisms

### Transformers

- Hugging Face Transformers
- RoBERTa
- DistilRoBERTa

### Deployment

- Streamlit
- Hugging Face Hub

---

# ⚠️ Limitations

Despite strong overall performance, the model has several limitations:

- **Neutral sentiment is the most difficult class.**
- Ambiguous reviews can be difficult to classify.
- Mixed-sentiment reviews may contain both positive and negative signals.
- Very short reviews may provide limited contextual information.
- Informal language and spelling variations can affect predictions.
- Softmax confidence should not automatically be interpreted as calibrated probability.
- Transformer models require considerably more computational resources than traditional ML models.

---

# 🚀 Future Improvements

Possible future improvements include:

- Fine-tuning larger Transformer architectures
- Improving Neutral-class recognition
- Hyperparameter optimization
- Data augmentation for difficult classes
- Confidence calibration
- Ensemble models
- Explainable AI techniques
- Detailed error analysis
- Model quantization for faster inference
- Further deployment optimization

---

# 💻 Run Locally

Clone the repository:

```bash
git clone https://github.com/jitheshpoojary/sentiment-analysis.git
cd sentiment-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application downloads the trained RoBERTa model from Hugging Face when required.

---

# ⭐ Key Highlights

```text
✓ 600K+ Amazon Reviews
✓ 3-Class Sentiment Classification
✓ Complete NLP Pipeline
✓ Machine Learning Experiments
✓ Deep Learning Experiments
✓ Transformer Fine-Tuning
✓ RoBERTa-base Final Model
✓ ~91% Accuracy
✓ ~80% Macro F1
✓ Hugging Face Model Hosting
✓ Interactive Streamlit Deployment
```

---

# 👨‍💻 Author

## Jithesh J Poojary

**Aspiring Data Scientist | Machine Learning Enthusiast**

<p align="left">
  <a href="https://github.com/jitheshpoojary">
    <img src="https://img.shields.io/badge/GitHub-jitheshpoojary-181717?style=flat-square&logo=github" alt="GitHub">
  </a>
  <a href="https://huggingface.co/jitheshpoojary">
    <img src="https://img.shields.io/badge/Hugging%20Face-jitheshpoojary-FFD21E?style=flat-square&logo=huggingface" alt="Hugging Face">
  </a>
</p>

---

<p align="center">
  ⭐ If you found this project useful, consider giving the repository a star!
</p>
