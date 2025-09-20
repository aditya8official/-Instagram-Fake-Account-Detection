# 📸 Instagram Fake vs Genuine Account Detection

## 📖 Overview
Fake and spam accounts are a big problem on social media platforms like Instagram.  
This project uses **Machine Learning (ML)** and **Data Analysis (DA)** techniques to classify Instagram accounts as **Fake** or **Genuine** based on profile features (followers, posts, bio, etc.).  

---

## 📂 Dataset
- **Training Dataset**: 576 rows, 12 columns  
- **Testing Dataset**: 120 rows, 13 columns  
- **Features**:  
  - Profile picture (binary)  
  - Username length and digits  
  - Full name length/words  
  - Bio/description length  
  - Private/Public status  
  - Posts, Followers, Following  
  - Target: Fake (1) or Genuine (0)  

---

## ⚙️ Tech Stack
- **Language**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, python-docx  
- **Tools**: Tableau (for visualization), SQL (for structured analysis), Excel  

---

## 🛠️ Project Workflow
1. **Data Preparation**  
   - Load training & testing datasets  
   - Handle missing values  
   - Feature engineering (Follower/Following ratio)  

2. **Exploratory Data Analysis (EDA)**  
   - Distribution of fake vs genuine accounts  
   - Correlation heatmaps  
   - Boxplots (followers, follows, posts vs fake/genuine)  

3. **Data Preprocessing**  
   - Scaling/normalization  
   - Feature selection  

4. **Model Building**  
   - Decision Tree Classifier  
   - Random Forest Classifier  

5. **Model Evaluation**  
   - Accuracy, Precision, Recall, F1-score  
   - Confusion Matrix & Visualization  
   - Feature Importance analysis  

6. **Predictions & Deployment**  
   - Save model with **Joblib**  
   - Test on unseen dataset  
   - Export predictions (`test_predictions.csv`)  

---

## 📊 Results
- **Decision Tree Accuracy**: ~90.5%  
- **Random Forest Accuracy**: ~92.2%  
- **Best Model**: Random Forest  

**Confusion Matrix (Random Forest)**:  
```
[[52  6]
 [ 3 55]]
```

**Classification Report**:  
```
              precision    recall  f1-score   support
           0       0.95      0.90      0.92        58
           1       0.90      0.95      0.92        58
    accuracy                           0.92       116
   macro avg       0.92      0.92      0.92       116
weighted avg       0.92      0.92      0.92       116
```

---

## 🔮 Future Improvements
- Use **XGBoost / Gradient Boosting** for better accuracy  
- Handle class imbalance with **SMOTE**  
- Deploy as a **Flask/Django web app** for real-time detection  
- Integrate **social network features** (friend connections, engagement metrics)  

---

## 📎 Repository Structure
```
📂 Instagram-Fake-Account-Detection
 ┣ 📂 data
 ┃ ┣ train.csv
 ┃ ┣ test.csv
 ┣ 📂 models
 ┃ ┣ rf_instagram_fake_model.joblib
 ┣ 📂 results
 ┃ ┣ test_predictions.csv
 ┃ ┣ confusion_matrix.png
 ┃ ┣ feature_importance.png
 ┣ main.py
 ┣ report_generator.py
 ┣ requirements.txt
 ┣ README.md
 ┗ LICENSE
```

---

## ▶️ How to Run
1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/Instagram-Fake-Account-Detection.git
   cd Instagram-Fake-Account-Detection
   ```
2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script  
   ```bash
   python main.py
   ```

---

## 📜 License
This project is licensed under the MIT License.  
