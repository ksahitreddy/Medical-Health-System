import streamlit as st
from llm_local1 import query_local_llm

# Streamlit UI
def app():
    st.title("🩺 Medical Chatbot")
    st.image('./images/capsule.png')
    st.success("Please ask your medical health problems and queries")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_query = st.text_input("Ask your question:")

    if st.button("Get Answer"):
        if user_query:
            response = query_local_llm(user_query)
            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("Chatbot", response))

    # Display chat history
    st.subheader("Chat History:")
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑‍⚕️ {role}:** {message}")
        else:
            st.markdown(f"**🤖 {role}:** {message}")

# Run the chatbot
if __name__ == "__main__":
    app()
