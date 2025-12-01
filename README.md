# ⛑️ CrisisResponse AI: Intelligent Humanitarian Triage

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://crisis-response-ai-hottam-ud-din.streamlit.app)
![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Deployed-success.svg)

**A Machine Learning pipeline for automating the classification of disaster messages to accelerate humanitarian aid.**

---

## 🔗 Quick Links
* **🔴 Live Dashboard:** [Click here to view the App](https://crisis-response-ai-hottam-ud-din.streamlit.app)
* **📂 Original Dataset:** [Kaggle: Disaster Response Messages](https://www.kaggle.com/datasets/sidharth178/disaster-response-messages/data)

---

## 📖 Project Overview
During large-scale disasters (earthquakes, floods, conflict), aid organizations are overwhelmed by the velocity of incoming data. Field reports, social media SOS signals, and direct messages come in by the thousands—often in unstructured formats.

**The Problem:** Manually reading and routing these messages to the correct department (e.g., sending a "water" request to the WASH team) is slow and prone to human error.

**The Solution:** CrisisResponse AI is an end-to-end Machine Learning pipeline that ingests raw text and automatically tags it into **36 specific humanitarian clusters** (including Medical, Shelter, Food, Search & Rescue). This allows for real-time identification of needs and automated routing of resources.

---

## 🛠️ Key Features
* **Multi-Output Classification:** Utilizes a **Random Forest** model wrapped in a Multi-Output Classifier to predict multiple tags for a single message (e.g., a message can be flagged as both `Urgent` and `Medical_Help`).
* **Automated Geotagging:** Includes a rule-based NLP extraction layer to identify affected provinces (focus on Pakistan: KPK, Sindh, Punjab, Balochistan).
* **Interactive Command Center:** A Streamlit-based dashboard designed for field managers to test messages and visualize historical data trends.
* **ETL Pipeline:** Automated data cleaning scripts that process raw CSVs into a structured SQLite database.

---

## 📊 The Dataset
The model is trained on the **Disaster Response Messages** dataset provided by Figure Eight (Appen).
* **Size:** 26,248 labeled messages.
* **Source:** Real messages sent during disasters such as the 2010 Pakistan Floods, the Haiti Earthquake, and Superstorm Sandy.
* **Labels:** 36 binary categories (0 or 1) covering all major humanitarian clusters.

---

## 💻 Technical Architecture
The project is structured into three main components:

### 1. ETL Pipeline (`process_data.py`)
* Loads the `messages` and `categories` datasets.
* Cleans the category data (converts to binary).
* Handles duplicates and missing values.
* Saves the clean data to an SQLite database (`DisasterResponse.db`).

### 2. ML Pipeline (`train_classifier.py`)
* Loads data from the SQLite database.
* **Preprocessing:** Tokenization, Lemmatization, and Stop Word removal using NLTK.
* **Vectorization:** Converts text to numbers using `CountVectorizer` (Top 5000 features) and `TfidfTransformer`.
* **Modeling:** Trains a **Multi-Output Random Forest Classifier**.
* **Export:** Saves the trained model as a pickle file (`classifier.pkl`).

### 3. Web Application (`app.py`)
* Built with **Streamlit** and **Plotly**.
* Visualizes dataset statistics (imbalance, distribution).
* Provides a user interface for real-time message classification.

---

## 🚀 How to Run Locally

If you want to run this project on your own machine:

**1. Clone the Repository**
```bash
git clone [https://github.com/HottamUdDin/crisis-response-ai.git](https://github.com/HottamUdDin/crisis-response-ai.git)
cd crisis-response-ai
````

**2. Install Dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the App**

```bash
streamlit run app.py
```

*(Optional) To retrain the model:*

```bash
python process_data.py
python train_classifier.py
```

-----

## 📈 Model Performance

The Random Forest model provides robust performance across major categories. Due to class imbalance (some categories like 'Water' are rare), the pipeline prioritizes **Precision** to minimize false alarms in critical categories.

  * **F1-Score (Weighted):** \~0.93
  * **Top Performing Categories:** Related, Aid\_Related, Weather\_Related.

-----

## 👤 Author

**Hottam Ud Din** *Final Year BS Data Science Student* *Aspiring Humanitarian Data Scientist*

Developed with a focus on MERL (Monitoring, Evaluation, Research, and Learning) technologies.

```
```
