# 🩺 Breast Cancer Detection System

An end-to-end Machine Learning application that classifies breast tumors as **Malignant** or **Benign** from clinical features, designed to assist medical professionals in early diagnosis. The model reaches **96.5% test accuracy** on the Wisconsin Breast Cancer Dataset.

Early detection is critical in healthcare. This project turns complex clinical data into actionable insights through a fast REST API and an interactive dashboard.

---

## 🧠 Key Highlights

- **Machine Learning Core:** Random Forest and XGBoost classifiers, trained and fine-tuned on the Wisconsin Breast Cancer Dataset (30 clinical features).
- **Robust Backend:** A scalable **FastAPI** REST backend that handles real-time inference requests.
- **Interactive Frontend:** A **Streamlit** user interface with animated probability gauges and confidence charts built with **Plotly**.

---

## 📊 Performance Metrics

| Metric         | Score |
|----------------|-------|
| Test Accuracy  | 96.5% |
| Precision      | 97%   |
| Recall         | 96%   |

**Dataset:** Wisconsin Breast Cancer Dataset, 569 patient records across 30 clinical features.

---

## 🛠️ Tech Stack

| Area              | Tools                                   |
|-------------------|-----------------------------------------|
| Language          | Python                                  |
| Machine Learning  | Scikit-learn, XGBoost, Random Forest    |
| Backend           | FastAPI                                 |
| Frontend          | Streamlit, Plotly                       |
| Model Persistence | Joblib                                  |

---

## 📁 Project Structure

> Adjust the file names below to match your project.

```
breast-cancer-detection/
├── api.py                 # FastAPI backend (inference endpoints)
├── app.py                 # Streamlit frontend
├── train.py               # Model training and evaluation
├── model.pkl              # Trained model (saved with Joblib)
├── scaler.pkl             # Feature scaler (if used)
├── requirements.txt       # Python dependencies
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Or manually:

   ```bash
   pip install scikit-learn xgboost fastapi uvicorn streamlit plotly joblib pandas numpy
   ```

---

## 🚀 Usage

### 1. Start the FastAPI backend

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs are available at http://localhost:8000/docs.

### 2. Start the Streamlit frontend

In a second terminal:

```bash
streamlit run app.py
```

Open http://localhost:8501, enter the 30 clinical feature values, and view the prediction with its probability gauge and confidence chart.

---

## 🔌 API Overview

> Update the routes and fields to match your backend.

| Method | Endpoint   | Description                                      |
|--------|------------|--------------------------------------------------|
| GET    | `/health`  | Health check                                     |
| POST   | `/predict` | Send the 30 clinical features, get a prediction  |

**Example response**

```json
{
  "prediction": "Benign",
  "probability_malignant": 0.04,
  "probability_benign": 0.96
}
```

---

## 🔄 ML Pipeline

1. Load the Wisconsin Breast Cancer Dataset (569 records, 30 features)
2. Preprocess and split the data into training and test sets
3. Train and fine-tune Random Forest and XGBoost classifiers
4. Evaluate with accuracy, precision and recall
5. Save the best model with Joblib
6. Serve predictions through FastAPI and visualize them in Streamlit with Plotly

---

## 🔮 Future Improvements

- Model explainability (for example SHAP feature importance)
- Cross-validation and further hyperparameter tuning
- Docker support and cloud deployment
- Validation on larger and more diverse clinical datasets

---

## ⚠️ Medical Disclaimer

This project is for **educational and research purposes only**. It is **not** a medical device and must not be used as a substitute for professional medical diagnosis, advice or treatment. Always consult a qualified healthcare professional.

---

## 💬 Feedback

Thoughts, questions or suggestions about the pipeline? Feel free to open an issue or reach out.

---

## 📄 License

Add your preferred license here (for example MIT).
