import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import plotly.express as px
import nltk

# --- NLTK SETUP ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download(['punkt', 'wordnet', 'omw-1.4', 'punkt_tab'], quiet=True)

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Disaster Response Dashboard | Hottam Ud Din",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM PROFESSIONAL CSS (FORCE LIGHT MODE) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    /* 1. FORCE BACKGROUNDS */
    .stApp {
        background-color: #f8f9fa; /* Light Gray */
    }
    
    /* 2. TEXT COLORS */
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: #2c3e50 !important; /* Dark Blue-Gray */
    }
    
    /* 3. CARD STYLING */
    .dashboard-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
    
    /* 4. METRICS FIX */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e9ecef;
        color: #000000 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #6c757d !important; /* Muted Text */
    }
    div[data-testid="stMetricValue"] {
        color: #2c3e50 !important; /* Dark Text */
    }
    
    /* 5. INPUT AREAS */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        border: 1px solid #ced4da;
    }
    
    /* 6. HEADER STYLING */
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a !important; /* Deep Blue */
        margin-bottom: 5px;
    }
    .header-subtitle {
        font-size: 1.1rem;
        color: #64748b !important;
        margin-bottom: 20px;
    }
    .author-tag {
        background-color: #e0f2fe;
        color: #0284c7 !important;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        display: inline-block;
        margin-bottom: 20px;
        border: 1px solid #bae6fd;
    }
    
    /* 7. REMOVE HEADER/FOOTER */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 8. BADGES */
    .badge-urgent { 
        background-color: #fecaca; 
        color: #b91c1c !important; 
        padding: 4px 8px; 
        border-radius: 4px; 
        font-weight: 600; 
    }
    .badge-normal { 
        background-color: #e0f2fe; 
        color: #0369a1 !important; 
        padding: 4px 8px; 
        border-radius: 4px; 
        font-weight: 600; 
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def detect_province(text):
    text = str(text).lower()
    if any(x in text for x in ['peshawar', 'kpk', 'swat', 'khyber']): return 'KPK'
    elif any(x in text for x in ['sindh', 'karachi', 'sukkur']): return 'Sindh'
    elif any(x in text for x in ['punjab', 'lahore', 'multan']): return 'Punjab'
    elif any(x in text for x in ['balochistan', 'quetta']): return 'Balochistan'
    return "Global / Unknown"

# --- LOAD DATA ---
@st.cache_resource
def load_model():
    with open('classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('disaster_response_table', engine)
    return df

model = load_model()
df = load_data()

# --- HEADER SECTION ---
st.markdown(f"""
    <div>
        <div class="header-title">Disaster Response Classification Dashboard</div>
        <div class="header-subtitle">Multi-Label Classification Analysis on 26,000+ Disaster Response Messages</div>
        <div class="author-tag">Developed by Hottam Ud Din</div>
    </div>
""", unsafe_allow_html=True)

# --- TOP METRICS ROW ---
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Total Dataset Size", f"{df.shape[0]:,}", "Messages")
with m2: st.metric("Humanitarian Categories", "36", "Classes")
with m3: st.metric("Model Architecture", "Random Forest", "Multi-Output")
with m4: st.metric("Training Status", "Completed", "Ready")

st.markdown("---")

# --- MAIN DASHBOARD GRID ---
col_left, col_right = st.columns([1, 2])

# === LEFT COLUMN: INTERACTIVE CLASSIFIER ===
with col_left:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("#### 🤖 Model Interactive Demo")
    st.markdown("Test the classification pipeline with custom text.")
    
    user_input = st.text_area(
        "Input Text",
        height=150,
        placeholder="E.g., 'We are trapped in Swat with no food and need urgent medical help.'",
        label_visibility="collapsed"
    )
    
    run_btn = st.button("Classify Message", type="primary", use_container_width=True)
    
    if run_btn and user_input:
        st.markdown("---")
        # Predict
        classification_labels = model.predict([user_input])[0]
        category_names = [c for c in df.columns if c not in ['id', 'message', 'original', 'genre', 'province_mentioned']]
        classification_results = dict(zip(category_names, classification_labels))
        active_tags = [k for k, v in classification_results.items() if v == 1]
        loc = detect_province(user_input)

        st.markdown(f"**Detected Region:** `{loc}`")
        
        if active_tags:
            st.markdown("**Predicted Categories:**")
            html_badges = ""
            for tag in active_tags:
                display_text = tag.replace('_', ' ').title()
                if tag in ['medical_help', 'water', 'food', 'shelter', 'death']:
                    html_badges += f'<span class="badge-urgent">{display_text}</span> '
                else:
                    html_badges += f'<span class="badge-normal">{display_text}</span> '
            st.markdown(html_badges, unsafe_allow_html=True)
        else:
            st.info("No specific categories detected.")
            
    st.markdown('</div>', unsafe_allow_html=True)

    # ADDING DATASET SAMPLE
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("#### 📂 Dataset Sample")
    st.dataframe(df[['message', 'genre']].head(5), hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# === RIGHT COLUMN: ANALYTICS ===
with col_right:
    # 1. BAR CHART
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Category Distribution")
    
    # Calculate counts
    numeric_df = df.select_dtypes(include=[np.number])
    if 'id' in numeric_df.columns: numeric_df = numeric_df.drop('id', axis=1)
    top_needs = numeric_df.sum().sort_values(ascending=False).head(10)
    
    # Force Plotly White Theme
    fig_bar = px.bar(
        x=top_needs.values,
        y=top_needs.index,
        orientation='h',
        labels={'x': 'Number of Messages', 'y': 'Category'},
        template="plotly_white", # THIS IS KEY FOR LIGHT MODE
        color=top_needs.values,
        color_continuous_scale='Blues'
    )
    fig_bar.update_layout(
        yaxis=dict(autorange="reversed"), 
        height=350, 
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
        plot_bgcolor='rgba(0,0,0,0)'
    )
    # Force font color to dark
    fig_bar.update_layout(font=dict(color="#2c3e50"))
    
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. PIE CHART ROW
    r1, r2 = st.columns(2)
    
    with r1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("#### 📡 Message Sources")
        genre_counts = df['genre'].value_counts()
        fig_pie1 = px.pie(
            values=genre_counts.values,
            names=genre_counts.index,
            hole=0.5,
            template="plotly_white",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig_pie1.update_layout(
            height=250, 
            margin=dict(t=20, b=20, l=20, r=20), 
            showlegend=False,
            font=dict(color="#2c3e50")
        )
        st.plotly_chart(fig_pie1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("#### 🗺️ Geographic Mentions")
        if 'province_mentioned' in df.columns:
            loc_counts = df['province_mentioned'].value_counts()
            loc_counts = loc_counts[loc_counts.index != 'Unknown']
            
            if not loc_counts.empty:
                fig_pie2 = px.pie(
                    values=loc_counts.values,
                    names=loc_counts.index,
                    hole=0.5,
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_pie2.update_layout(
                    height=250, 
                    margin=dict(t=20, b=20, l=20, r=20), 
                    showlegend=False,
                    font=dict(color="#2c3e50")
                )
                st.plotly_chart(fig_pie2, use_container_width=True)
            else:
                st.info("No locations found.")
        st.markdown('</div>', unsafe_allow_html=True)