# Advanced Machine Learning Project - DSC 540
## Human Activity Recognition Using mHealth Sensor Data

### 📋 Project Overview
This project implements machine learning algorithms for human activity recognition using wearable sensor data from the mHealth dataset. The goal is to classify 12 different physical activities using multi-sensor data from chest, ankle, and arm-mounted devices.

### 🎯 Objective
Develop and compare machine learning models to accurately classify human activities based on accelerometer, gyroscope, and magnetometer sensor readings.

### 📊 Dataset
- **Source**: mHealth Dataset
- **Participants**: 10 subjects
- **Activities**: 12 different physical activities
- **Sensors**: 3 body-mounted sensors (chest, left ankle, right arm)
- **Features**: 23 sensor measurements per sample
- **Sampling Rate**: 50Hz
- **Total Samples**: 1,215,755 data points

#### Activity Classes:
1. Standing still
2. Sitting and relaxing  
3. Lying down
4. Walking
5. Climbing stairs
6. Waist bends forward
7. Frontal elevation of arms
8. Knees bending
9. Cycling
10. Jogging
11. Running
12. Jump front & back

### 🛠️ Methodology

#### Data Preprocessing:
- Combined data from all 10 subjects
- Removed null activity samples (Label = 0)
- Applied sliding window technique (100 samples, 50% overlap)
- Extracted statistical features (mean, std, min, max, RMS)

#### Feature Engineering:
- Window-based feature extraction
- 115 total features from 23 sensor channels
- Feature selection using Random Forest and XGBoost importance
- Reduced to 29 most important features (74.8% reduction)

#### Models Implemented:
- **Random Forest Classifier**
- **XGBoost Classifier**
- Feature importance analysis
- SHAP explainability analysis

### 📈 Results
- Achieved high accuracy in activity classification
- Successful feature reduction without performance loss
- Identified key sensor features for activity recognition
- Cross-validation confirmed model robustness

### 🔧 Technologies Used
- **Python 3.x**
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting
- **SHAP** - Model explainability
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization

### 📁 Project Structure

├── data/ # Sensor data files
│ ├── mHealth_subject1.csv # Subject 1 data
│ ├── mHealth_subject2.csv # Subject 2 data
│ └── ... # Additional subject files
├── Mhealth.ipynb # Main analysis notebook
├── Jainilkumar_AML_PROJECT.pdf # Project report
├── Jainil_Project_Proposal.pdf # Project proposal
└── README.md # This file


### 🚀 Getting Started

#### Prerequisites
```bash
pip install pandas scikit-learn xgboost shap numpy matplotlib seaborn scipy
```

#### Running the Analysis
1. Clone this repository
2. Install required packages
3. Open `Mhealth.ipynb` in Jupyter Notebook/Lab
4. Run all cells to reproduce the analysis

### 🔍 Key Findings
- Ankle sensors showed the highest importance for activity classification
- Statistical features (mean, RMS) were most discriminative
- Feature selection maintained performance while reducing complexity
- Both Random Forest and XGBoost achieved comparable results

### 👤 Author
**Jainil Kumar**  
DePaul University - DSC 540 Advanced Machine Learning

### 📄 License
This project is for academic purposes as part of the DSC 540 course.


*For detailed methodology and results, please refer to the project report and Jupyter notebook.*
