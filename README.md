# 📰 Fake News Detection Website

A **Machine Learning + NLP based web application** that detects whether a given news article is **REAL**, **FAKE**, or **UNCERTAIN**. The project is built as a **Final Year Mini Project** using Python and Streamlit.

---

## 🚀 Features

* 🔍 **News Prediction** (Real / Fake / Uncertain)
* 📊 **Model Analytics**

  * Accuracy
  * Confusion Matrix
  * ROC Curve
* 🕒 **Prediction History** (session-based)
* 🗺️ **Country-wise Map Visualization**
* 🧹 Robust text cleaning & preprocessing
* 🌐 Interactive **Streamlit Web Interface**

---

## 🧠 Machine Learning Details

* **Algorithm**: Multinomial Naive Bayes
* **Text Vectorization**: TF-IDF
* **Evaluation Metrics**:

  * Accuracy
  * Confusion Matrix
  * ROC Curve & AUC

---

## 📂 Dataset Used

* **BBC News Dataset** → Real news
* **Fake News Dataset** → Fake news

> ⚠️ Datasets and trained model files are excluded from GitHub using `.gitignore` (best practice).

---

## 🗂️ Project Structure

```
fake-news-detection/
│── app.py                # Streamlit web app
│── train_model.py        # Model training & evaluation
│── .gitignore            # Ignored files & folders
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Anand3420/fake-news-detection.git
cd fake-news-detection
```

### 2️⃣ Create virtual environment (optional but recommended)

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the model

```bash
python train_model.py
```

### 5️⃣ Run the web app

```bash
streamlit run app.py
```

---

## 🌐 Web Interface Tabs

* **📰 Prediction** – Enter news text and get prediction
* **📊 Analytics** – Accuracy, Confusion Matrix, ROC Curve
* **🕒 History** – Past predictions
* **🗺️ Map** – Country-wise news visualization
* **ℹ️ About** – Project details

---

## 📌 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Git & GitHub

---

## 🎓 Academic Use

This project is suitable for:

* Final Year Mini Project
* Machine Learning / NLP coursework
* Resume & Portfolio projects

---

## 👨‍💻 Developer

* **Name**: Anand
* **Year**: 2025
* **Project Type**: Final Year Mini Project

---

## 📜 License

This project is for **educational purposes only**.

---

⭐ *If you like this project, give it a star on GitHub!*
