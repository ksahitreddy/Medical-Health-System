import streamlit as st
from web_functions import load_data

from Tabs import diagnosis, home, talk2doc

# Configure the app
st.set_page_config(
    page_title = 'Medical Assistance System',
    page_icon = '🥯',
    layout = 'wide',
    initial_sidebar_state = 'auto'
)

Tabs = {
    "Home":home,
    "Ask Queries":talk2doc,
    "Diagnosis":diagnosis,
}

st.sidebar.title('Navigation')

page = st.sidebar.selectbox("Page", list(Tabs.keys()))

df, X, y = load_data()

if page in ["Diagnosis"]:
    Tabs[page].app(df, X, y)
else:
    Tabs[page].app()
