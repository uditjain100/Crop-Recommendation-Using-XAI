# 🌾 Use of Explainable AI using SHAP on Crop Yield Prediction in India

Agriculture remains the backbone of the Indian economy, employing more than half of the country's population. However, the sector faces several challenges including unpredictable weather patterns, changing soil health, and fluctuating crop production. Accurate crop yield prediction is essential for ensuring food security, planning agricultural activities, and formulating government policies. Traditional models often struggle to handle the complexity and non-linearity of such data. This project leverages modern machine learning techniques to create a high-accuracy crop yield prediction system tailored for Indian conditions.

What makes this project stand out is the integration of Explainable AI (XAI) techniques—specifically, SHAP (SHapley Additive exPlanations)—into the model pipeline. While models like Random Forest and XGBoost offer strong predictive capabilities, they often act as "black boxes." SHAP enhances interpretability by quantifying how much each feature contributes to a prediction. This empowers not only data scientists but also farmers, agricultural officers, and policymakers to understand the ‘why’ behind each yield forecast and make informed, trust-based decisions.

### 🔑 Key Highlights of the Project

- Built a complete machine learning pipeline for crop yield prediction using Indian agricultural datasets.
- Applied feature engineering to clean, encode, and scale categorical and numerical variables.
- Trained and evaluated multiple ML models including **Random Forest**, **XGBoost**, and **Decision Tree**.
- Achieved high accuracy with **Random Forest** (98.96%) and minimal error metrics (MAE: 1.97, RMSE: 2.45).
- Integrated **SHAP** to interpret both **global** feature importance and **local** prediction rationale.
- Visualized feature contributions using SHAP’s **summary**, **force**, **waterfall**, and **decision** plots.
- Identified top influential features such as **crop type**, **cultivated area**, and **seasonal variations**.
- Used SHAP to explain anomalous predictions and outliers, aiding trust and model debugging.
- Created a reproducible and modular codebase structured around data preprocessing, modeling, and XAI.
- Organized the project into well-documented Jupyter notebooks and visual asset directories.
- Added future-ready extensibility for incorporating **weather**, **satellite imagery**, or **real-time IoT sensor data**.
- Positioned the project for potential **mobile app deployment** to assist farmers with yield insights in regional languages.

This project presents a robust and interpretable machine learning pipeline that predicts agricultural crop yield in India using various ML models, including **Random Forest**, **XGBoost**, and **Decision Tree**. The system integrates **Explainable AI (XAI)** through **SHAP (SHapley Additive exPlanations)** to provide transparent insights into model decisions. The project is designed for both technical and non-technical stakeholders like policymakers and farmers.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [How to Run](#how-to-run)
- [Model Details](#model-details)
- [Explainability with SHAP](#explainability-with-shap)
- [Results and Visualizations](#results-and-visualizations)
- [Research References](#research-references)
- [Future Work](#future-work)

---

## ✅ Overview

India's economy is deeply rooted in agriculture, and predicting crop yields is essential for food security, planning, and policy-making. While machine learning (ML) models provide high accuracy, their black-box nature limits interpretability. This project bridges that gap by integrating **SHAP** for **explainable AI (XAI)**.

### Objectives:

- Build a yield prediction system using Indian agricultural data
- Train and compare Decision Tree, XGBoost, and Random Forest
- Apply SHAP to make models transparent and interpretable
- Visualize feature contributions to enhance trust and usability

---

## 🌟 Key Features

- Predicts crop yield based on multiple features: crop type, area, season, etc.
- Uses **Random Forest**, **Decision Tree**, and **XGBoost** models
- Integrates SHAP for model interpretability
- Provides **visual insights** using SHAP plots: summary, force, waterfall, and decision plots
- Highlights actionable patterns (e.g., influence of crop type and region on yield)

---

## 📁 Project Structure

```
├── code/
│   ├── crop_recom_saf_code/
│   │   ├── assets/
│   │   ├── plantuml-imp/
│   │   ├── py-imp/
│   │   │   ├── code/
│   │   │   ├── models/
│   │   │   ├── pump/
│   │   ├── r-imp/
│   │   └── scratch/
│   └── crop_yield_code/
│       ├── code/
│       ├── dl_code/
│       ├── dataset/
│       └── notebooks/
├── dataset/
├── documentation/
│   ├── Minor_Project_report.pdf
│   ├── summary.png, force.png, decision.png, waterfall.png (Latex_minor/)
├── researchpapaers/
│   ├── SHAP_Paper.pdf
│   ├── LIME_Paper.pdf
│   └── Other Related Work
└── README.md
```

```
crop-yield-shap/
│
├── data/                     # CSV dataset containing crop yield records (cleaned & encoded)
├── models/                   # Trained models (Pickle/Joblib format)
├── notebooks/                # Jupyter notebooks for EDA, training, and SHAP analysis
├── shap_plots/               # SHAP visualization outputs (summary, force, etc.)
├── src/
│   ├── preprocessing.py      # Data loading and preprocessing
│   ├── train_model.py        # Model training scripts
│   ├── shap_analysis.py      # SHAP value computation and plotting
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── main.py                   # Run the full pipeline
```

---

## ⚙️ Prerequisites

To run this project, ensure you have the following environment and dependencies properly set up:

### 🐍 Python Environment

- **Python Version:** 3.7 or higher is required.
  - Recommended: Use a virtual environment (`venv` or `conda`) to manage dependencies and avoid version conflicts.

### 📦 Package Manager

You can use either:

- `pip` (Python's standard package installer)
- `conda` (Anaconda distribution, optional but convenient for managing environments)

---

## 🔧 Installation Steps

1. **Clone the repository** (if hosted on GitHub):

```bash
git clone https://github.com/yourusername/crop-yield-shap.git
cd crop-yield-shap
```

2. **Create and activate a virtual environment:**

Using `venv`:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or using `conda`:

```bash
conda create --name shapenv python=3.8
conda activate shapenv
```

3. **Install the required Python libraries:**

```bash
pip install -r requirements.txt
```

---

## 📄 Contents of `requirements.txt`

These are the core Python libraries used in this project:

```txt
pandas             # For data loading and manipulation
numpy              # For numerical operations
scikit-learn       # For ML models like Decision Tree and Random Forest
xgboost            # For the XGBoost model
shap               # For SHAP (SHapley Additive exPlanations) interpretability
matplotlib         # For basic plotting
seaborn            # For enhanced visualization
jupyter            # To run the interactive .ipynb notebooks
```

> Make sure you have JupyterLab or Notebook installed to run the `.ipynb` files interactively. SHAP visualizations also require a supported browser interface.

If you face any issues with library versions, try upgrading pip:

```bash
pip install --upgrade pip
```

Or install libraries individually.

- Python 3.7 or higher
- pip or conda for package management

### Install Required Libraries

```bash
pip install -r requirements.txt
```

### Contents of `requirements.txt`

```txt
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
jupyter
```

---

## ▶️ How to Run

To run the entire pipeline:

```bash
python main.py
```

Alternatively, run step-by-step in Jupyter:

```bash
jupyter notebook
```

Open `notebooks/Crop_Yield_Prediction.ipynb` to interactively run preprocessing, training, and SHAP analysis.

---

## 🤖 Model Details

This project explores and evaluates three major machine learning models—**Decision Tree**, **XGBoost**, and **Random Forest**—to predict agricultural crop yield using a dataset curated from Indian states. Each model was trained on preprocessed data that includes both numerical (e.g., area, production) and categorical features (e.g., crop type, season, state). Performance was evaluated using standard regression metrics including **Accuracy**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**.

### 1. 🌿 **Decision Tree Regressor**

The Decision Tree model works by learning hierarchical rules from data, splitting based on feature thresholds to reduce prediction error at each level. It is a **non-parametric**, **interpretable** model that forms the foundation for many ensemble methods.

- **Advantages**: Easy to interpret, visualizable, and works well on small to medium datasets.
- **Limitations**: Prone to overfitting; performance can degrade with noisy or unbalanced data.
- **Performance**:
  - **Accuracy**: ~89.78%
  - **MAE**: ~5.16
  - **RMSE**: ~6.12

### 2. ⚙️ **XGBoost Regressor**

XGBoost (Extreme Gradient Boosting) is an advanced ensemble model that builds multiple weak learners (decision trees) sequentially. It optimizes for speed and accuracy using a gradient boosting framework with regularization, making it highly effective on structured data.

- **Advantages**: Robust to outliers and missing values, handles non-linear relationships well, supports feature importance ranking.
- **Limitations**: Difficult to interpret without explainability tools like SHAP, longer training time compared to simpler models.
- **Performance**:
  - **Accuracy**: ~86.46%
  - **MAE**: ~4.88
  - **RMSE**: ~5.71

### 3. 🌲 **Random Forest Regressor (Best Performer)**

Random Forest is an ensemble model that constructs a multitude of decision trees during training and outputs the average of their predictions. It reduces variance and improves generalization by using bootstrapped datasets and random feature selection for each tree.

- **Advantages**: High accuracy, less prone to overfitting, naturally supports feature importance.
- **Limitations**: Less interpretable than a single decision tree, larger memory footprint.
- **Performance**:
  - **Accuracy**: **98.96%**
  - **MAE**: **1.97**
  - **RMSE**: **2.45**

### 🧪 Evaluation Summary

| Model             | Accuracy (%) | MAE      | RMSE     |
| ----------------- | ------------ | -------- | -------- |
| Decision Tree     | 89.78        | 5.16     | 6.12     |
| XGBoost           | 86.46        | 4.88     | 5.71     |
| **Random Forest** | **98.96**    | **1.97** | **2.45** |

📌 **Conclusion**: The **Random Forest Regressor** significantly outperforms the other models across all evaluation metrics and is selected for integration with SHAP for explainability.

Three machine learning models were implemented and evaluated:

### 1. **Decision Tree**

- Simple rule-based model
- Interpretability: High
- Accuracy: ~89.78%

### 2. **XGBoost**

- Gradient boosting with better error handling
- Accuracy: ~86.46%
- Less interpretable

### 3. **Random Forest (Best)**

- Ensemble of decision trees
- Accuracy: **98.96%**
- MAE: 1.97, RMSE: 2.45

---

## 🧠 Explainability with SHAP

A major strength of this project is its integration of **Explainable AI (XAI)** techniques using **SHAP (SHapley Additive exPlanations)**. While machine learning models—especially ensemble and boosting methods—can achieve high predictive accuracy, they often operate as "black boxes" with limited interpretability. SHAP helps open this black box by explaining **how** and **why** a model made a specific prediction.

### 🎯 Why Use SHAP?

- Rooted in **game theory**, SHAP assigns each feature a "Shapley value" representing its contribution to the model’s output.
- Works as a **model-agnostic** explainer—compatible with tree-based models (like Random Forest, XGBoost) as well as deep learning.
- Captures both:
  - **Global feature importance** (which features are most influential overall)
  - **Local explanations** (why a single prediction was made for a specific input)

### 📊 SHAP Visualizations Employed

Several SHAP visualizations were used to derive interpretability insights:

- ✅ **Summary Plot**: Visualizes the overall influence of each feature on the model's output across the dataset.
- 🔄 **Waterfall Plot**: Breaks down the impact of each feature on a single prediction starting from the base value.
- 🔍 **Force Plot**: Provides an interactive view of how individual features contribute positively or negatively to a prediction.
- 🧭 **Decision Plot**: Shows the accumulated contribution of features as the model makes splits or decisions.

These plots are integrated into the analysis notebooks for both ML and DL models, helping us debug, interpret, and validate model behavior from a domain perspective.

### 💡 Impact of SHAP in This Project

- Allowed clear identification of key features like `Crop Type`, `Area`, and `Season` driving the model’s predictions.
- Helped highlight cases where models may have **overfit** or made unexpected decisions.
- Strengthened trust and transparency in AI-driven recommendations for **non-technical users** like farmers and agricultural officers.

> SHAP acts as the bridge between complex AI models and real-world usability in agriculture.

**SHAP (SHapley Additive exPlanations)** assigns each feature a contribution score for a prediction.

### Why SHAP?

- Based on **game theory** (Shapley values)
- Model-agnostic
- Captures both **global** and **local** feature importance

### SHAP Visualizations Used

- **Summary Plot** – overall feature importance
- **Waterfall Plot** – contribution of features to one prediction
- **Force Plot** – visual explanation per instance
- **Decision Plot** – cumulative impact across decisions

SHAP helps interpret model behavior and supports domain experts in trusting the AI system.

---

## 📊 Results and Visualizations

After training and evaluating multiple machine learning models, the performance was assessed using standard regression metrics: **Accuracy**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**. The following table summarizes the model performance on the crop yield prediction task:

### 🔍 Model Evaluation Summary

| Model             | Accuracy (%) | MAE      | RMSE     |
| ----------------- | ------------ | -------- | -------- |
| Decision Tree     | 89.78        | 5.16     | 6.12     |
| XGBoost           | 86.46        | 4.88     | 5.71     |
| **Random Forest** | **98.96**    | **1.97** | **2.45** |

- **Accuracy**: Proportion of predictions close to actual yield values.
- **MAE**: Average absolute difference between predicted and actual yields.
- **RMSE**: Penalizes larger errors more than MAE; a lower value indicates higher model precision.

### 🔎 Key Visual Insights via SHAP

To understand how predictions are made, SHAP was used to visualize and rank the contribution of features. The following insights were derived from global and local SHAP plots:

#### 🌟 Top Influential Features

- **Crop Type**: Certain crops like Rice and Coconut showed significantly different yield patterns.
- **Area Under Cultivation**: Larger areas often correlate with higher yield, but non-linearly.
- **Season**: Seasons such as **Kharif** and **Rabi** influence irrigation availability and crop growth.
- **Region/District Variations**: Specific crop-region interactions (e.g., **Urad in TIRUNELVELI**) had strong local influence.

#### 🖼️ Visual Outputs Included

- **Confusion Matrices** for classification comparison (converted for interpretability in regression via binning).
- **Feature Importance Plots**: SHAP bar plots to rank features by mean absolute SHAP values.
- **Force and Waterfall Plots**: Per-instance explanations showing how individual features influenced the model.
- **Decision Plots**: Show how the cumulative effect of features leads to final predictions.

> These results not only highlight model performance but also guide agronomists and policymakers to understand _why_ a certain recommendation or yield estimate was made.

---

## 📚 Research References

- S. M. Lundberg & S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” NeurIPS 2017.
- Chlingaryan et al., “ML for Crop Yield and Nitrogen Estimation,” _Computers & Electronics in Agriculture_, 2018.
- Jeong et al., “Random Forests for Crop Yield,” _Agricultural and Forest Meteorology_, 2016.
- Barredo Arrieta et al., “Explainable AI: Taxonomies & Challenges,” _Information Fusion_, 2020.

> Full list available in `References` section of the Minor Project Report

---

## 🔮 Future Work

- Incorporate **real-time weather and satellite data**
- Deploy as a **mobile app** with region-wise recommendations in local languages
- Extend model to **village-level predictions**
- Explore **causal inference** alongside SHAP for better decision rationale
- Compress models for **low-resource devices** in rural settings

---

## 👩‍💻 Author

**Udit Jain**  
M.Tech, Roll No: 24CSM1R23  
Department of Computer Science and Engineering  
National Institute of Technology, Warangal  
Supervisor: **Dr. Manjubala Bisi**

---

## 🏁 Final Note

This project highlights how combining **machine learning** with **explainable AI** can solve real-world agricultural problems while keeping farmers and policymakers in the loop. SHAP makes the black box of AI transparent — empowering decisions that are both smart and trusted.
