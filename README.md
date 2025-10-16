# 🩺 Diabetes Risk Prediction – Machine Learning Project

## 📘 Overview
This project focuses on predicting **diabetes risk** using the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set).  
It walks through a complete **end-to-end Machine Learning workflow**, including data loading, exploratory analysis, preprocessing, modeling, and evaluation.

---

## 🧭 Project Workflow

### **Day 1 – Topic Selection & Data Acquisition**
- Chose diabetes prediction as the project topic.
- Downloaded dataset from Kaggle.
- Loaded data in `0_Data_load.ipynb`.
- Verified dataset structure and initial statistics.

### **Day 2 – Data Preparation & Cleaning**
- Conducted Exploratory Data Analysis (EDA) in `1_EDA.ipynb`.
  - Identified feature distributions, missing values, and correlations.
- In `2_Preprocessing.ipynb`:
  - Normalized features using **MinMaxScaler**.
  - Split data into **training** and **testing** sets (80/20).
  - Saved processed data to `/Data/Processed/`.

### **Day 3 – Model Development & Initial Tuning**
- Implemented **K-Nearest Neighbors (KNN)** classifier in `3_Modelling_Evaluation.ipynb`.
- Tuned `n_neighbors` hyperparameter to optimize accuracy.
- Evaluated with:
  - Confusion Matrix
  - Classification Report
  - Accuracy score

### **Day 4 – Advanced Modeling & Finalization**
- Compared KNN results with **Random Forest** and **Gradient Boosting**.
- Visualized feature importance and model performance.
- Prepared summary for final presentation.

---

## 🧩 Folder Structure

```
DIABETES-RISK-PREDICTION/
│
├── Data/
│   ├── Raw/
│   │   └── diabetes.csv
│   └── Processed/
│       ├── diabetes.csv
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── 0_Data_load.ipynb          # Data import & initial checks
├── 1_EDA.ipynb                # Exploratory Data Analysis
├── 2_Preprocessing.ipynb      # Data cleaning & feature scaling
├── 3_Modelling_Evaluation.ipynb  # Model training & evaluation
│
├── .gitignore
└── README.md
```

---

## 🧠 Dataset Details

| Feature | Description |
|----------|--------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Family diabetes history score |
| Age | Age (years) |
| Outcome | 0 = No Diabetes, 1 = Diabetes |

---

## ⚙️ Models Used

| Model | Accuracy | Remarks |
|--------|-----------|----------|
| **K-Nearest Neighbors (KNN)** | ~0.78 | Baseline model |
| **Random Forest** | ~0.83 | Better accuracy and stability |
| **Gradient Boosting** | ~0.84 | Best performing model |

---

## 📊 Visualizations

Generated in notebooks:
- Correlation Heatmap  
- Feature Distributions  
- Accuracy vs. K (for KNN)  
- Confusion Matrix  
- Feature Importance (Random Forest)

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/diabetes-risk-prediction.git
   cd diabetes-risk-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open notebooks in VS Code or Jupyter:**
   ```bash
   jupyter notebook
   ```
   and run notebooks in the order:
   - `0_Data_load.ipynb`
   - `1_EDA.ipynb`
   - `2_Preprocessing.ipynb`
   - `3_Modelling_Evaluation.ipynb`

---

## 🧩 Key Learnings
- Feature scaling significantly improves KNN performance.  
- Ensemble models generalize better than distance-based classifiers.  
- Glucose, BMI, Age, and Pregnancies are key predictors.  

---

## 🔮 Future Work
- Add **GridSearchCV** for systematic hyperparameter tuning.  
- Implement **cross-validation** for more robust evaluation.  
- Deploy best model using **Streamlit** or **Flask** for real-time prediction.

---

## 👩‍💻 Author
Varsha Maurya 
📧 varsha.eminent@gmail.com  
🔗 [[LinkedIn](https://www.linkedin.com/in/varsha-maurya/)](#) | [https://github.com/varsha199/Diabetes-Risk-Prediction-Using-Machine-Learning](#)
