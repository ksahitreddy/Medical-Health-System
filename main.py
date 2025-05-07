import streamlit as st
from web_functions_1 import load_data

from Tabs import diagnosis_1, home, talk2doc_1

# Configure the app
st.set_page_config(
    page_title = 'Medical Assistance System',
    page_icon = '🥯',
    layout = 'wide',
    initial_sidebar_state = 'auto'
)

Tabs = {
    "Home":home,
    "Ask Queries":talk2doc_1,
    "Diagnosis":diagnosis_1,
}

st.sidebar.title('Navigation')

page = st.sidebar.radio("Page", list(Tabs.keys()))

df, X, y = load_data()

if page in ["Diagnosis"]:
    Tabs[page].app(df, X, y)
else:
    Tabs[page].app()
