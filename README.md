🎓 EduPredict
An end-to-end Machine Learning project designed to predict student academic outcomes (Pass vs. At-Risk) using behavioral, academic, and lifestyle metrics. Powered by an optimized Random Forest Classifier, this project assists educators and institutions in identifying at-risk students early and planning timely interventions.

📊 Features & Attributes Tracked
The model analyzes 7 multi-dimensional student features:

Study_Hours_Weekly: Weekly hours dedicated to studying (2 to 35 hours).

Attendance_Pct: Overall classroom attendance percentage (40% to 100%).

Past_GPA: Previous academic score / Grade Point Average (3.0 to 10.0).

Sleep_Hours_Daily: Average daily sleep duration (4 to 10 hours).

Extracurricular: Participation in extracurricular activities (0 = No, 1 = Yes).

Family_Relations: Rating of family support and relations (1 to 5 scale).

Social_Engagement: Rating of social engagement and peer interaction (1 to 5 scale).

🛠️ Tech Stack & Libraries
Language: Python

Data Manipulation & Simulation: Pandas, NumPy

Machine Learning: Scikit-Learn (RandomForestClassifier, StandardScaler, train_test_split)

Model Persistence: Joblib

Visualization: Matplotlib, Seaborn

📂 Project Structure
Plaintext
student-performance-predictor/
│
├── dataset.ipynb                      # Jupyter / Colab notebook with end-to-end code
├── student_performance_dataset.csv    # Simulated student performance dataset
├── student_model.pkl                  # Trained Random Forest model (serialized)
├── feature_scaler.pkl                 # Fitted StandardScaler object (serialized)
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
🚀 Getting Started & Local Execution
1. Clone the Repository
Bash
git clone https://github.com/arti-nova/EduPredict
cd student-performance-predictor
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Load Model & Run Inference (Python Example)
Python
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('student_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Define a student profile: [Study_Hours, Attendance, Past_GPA, Sleep, Extracurricular, Family, Social]
new_student = [[8.5, 65.0, 5.5, 6.0, 0, 3, 2]]

# Scale and predict
new_student_scaled = scaler.transform(new_student)
prediction = model.predict(new_student_scaled)[0]
probabilities = model.predict_proba(new_student_scaled)[0]

status = "✅ On-Track (Pass)" if prediction == 1 else "⚠️ At-Risk (Needs Intervention)"
print(f"Prediction Result: {status}")
print(f"Confidence -> At-Risk: {probabilities[0]*100:.1f}% | Pass: {probabilities[1]*100:.1f}%")
📈 Model Performance & Evaluation
Algorithm: Random Forest Classifier (n_estimators=150, max_depth=8)

Validation Split: 80% Training, 20% Testing (Stratified)

Key Metrics: Evaluated using Accuracy Score, Classification Report (Precision/Recall/F1-Score), and Confusion Matrix visualizations.
