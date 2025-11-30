# ⛑️ CrisisResponse AI: Intelligent Humanitarian Triage

### 🚀 Project Overview
**CrisisResponse AI** is a machine learning pipeline designed to automate the classification of unstructured field data (SMS, reports, social media) during humanitarian crises.

In high-velocity disaster scenarios, manual tagging of thousands of incoming messages is a bottleneck. This tool ingests raw text and categorizes it into **36 specific humanitarian clusters** (e.g., WASH, Shelter, Medical, Food Security) to accelerate resource allocation.

### 🛠️ Key Features
* **Multi-Output Classification:** Uses a Random Forest Classifier to tag messages with multiple labels simultaneously (e.g., a message can be both "Urgent" and "Water Related").
* **Automated Geotagging:** NLP-based keyword extraction to identify affected regions (focus on Pakistan Flood zones).
* **Interactive Dashboard:** A Streamlit-based "Command Center" for field operators to input text and view real-time analytics.
* **Clean Pipeline Architecture:** Modularized ETL (Extract, Transform, Load) and ML training scripts.

### 📊 Tech Stack
* **Core:** Python 3.9, NumPy, Pandas
* **NLP:** NLTK (Tokenization, Lemmatization), Scikit-Learn (TF-IDF, Pipelines)
* **Visualization:** Plotly Express
* **Web App:** Streamlit

### 💻 How to Run
1. **Clone the repository**
   ```bash
   git clone [https://github.com/your-username/crisis-response-ai.git](https://github.com/your-username/crisis-response-ai.git)
````

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the ETL & ML Pipelines** (Optional - pre-trained models included)
    ```bash
    python process_data.py  # Cleans data to SQLite
    python train_classifier.py  # Trains and saves model.pkl
    ```
4.  **Launch the Dashboard**
    ```bash
    streamlit run app.py
    ```

### 📈 Model Performance

The model was trained on a dataset of **26,000+ disaster response messages**. It achieves an average F1-score of **0.93** across major categories (Related, Request, Aid\_Related), ensuring reliable triage for critical alerts.

-----

*Developed by **Hottam Ud Din** | Aspiring Data Scientist*

````

### Why this format works
* **The emoji in the title** makes it stand out.
* **The `###` headers** create clear sections that hiring managers can scan quickly.
* **The code blocks** (text inside ```) look professional and technical.

Once you save this file and upload it to GitHub, it will automatically appear as the "front page" of your repository.
````