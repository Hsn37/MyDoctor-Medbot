import os
import streamlit as st
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="MyDoctor Health Assistant",
    page_icon="🏥",
    layout="centered"
)

# Custom CSS with updated color scheme
st.markdown("""
<style>
    .main-header {
        color: #087484;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subheader {
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emergency-warning {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid #C62828;
        font-weight: bold;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #777;
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #ddd;
    }
    .stChatMessage .message-container .avatar.user {
        background-color: #ff9414 !important;
    }
    .stChatMessage .message-container .avatar.assistant {
        background-color: #087484 !important;
    }
    button[data-testid="chatInputSubmitButton"] {
        background-color: #087484 !important;
        color: white !important;
    }
    .sidebar .sidebar-content {
        background-color: #f5f9fa;
    }
    
    /* Make header visible */
    header[data-testid="stHeader"] {
        background-color: #087484 !important;
        border-bottom: 1px solid #f0f2f6;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* Navigation buttons in header */
    button[data-testid="baseButton-header"] {
        color: white !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* Navigation container */
    div[data-testid="stToolbar"] {
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    div[data-testid="stToolbar"] button {
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    div[data-testid="stChatInput"] {
        border-color: #087484;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #ff9414;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
    .logo {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background-color: #087484;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Simple navigation bar */
    .nav-bar {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin-bottom: 20px;
        padding: 10px;
        background-color: #f5f9fa;
        border-radius: 8px;
    }
    
    .nav-button {
        background-color: #087484;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        font-weight: bold;
    }
    
    .nav-button:hover {
        background-color: #ff9414;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Logo and App title
st.markdown('<div class="logo-container"><div class="logo">MD</div></div>', unsafe_allow_html=True)
st.markdown("<h1 class='main-header'>MyDoctor Health Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Your personal health guide</p>", unsafe_allow_html=True)


# Emergency warning
st.markdown("""
<div class="emergency-warning">
    ⚠️ If you are experiencing a medical emergency, please call emergency services immediately or go to your nearest emergency department.
</div>
""", unsafe_allow_html=True)

try:
    # Initialize RAG system
    @st.cache_resource(show_spinner=False)
    def get_rag_system():
        from rag_processor import UgandaMedicalRAG
        return UgandaMedicalRAG()

    # Display loading message while initializing
    if "rag" not in st.session_state:
        with st.spinner("Connecting to medical database..."):
            st.session_state.rag = get_rag_system()

    # Get RAG instance
    rag = st.session_state.rag
    
    # Display initial welcome message if no messages
    if len(st.session_state.messages) == 0:
        st.session_state.messages.append(
            {"role": "assistant", "content": "Hi there! I'm MyDoctor's digital health assistant. You can tell me how you're feeling, and I'll try to help. Tell me about the symptoms you're experiencing!"}
        )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your symptoms or health question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in the chat UI
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Create a proper conversation history from session state
                conversation_history = list(st.session_state.messages)
                
                # Stream the response using RAG
                for response_chunk in rag.generate_answer_stream(
                    query=prompt, 
                    message_history=conversation_history
                ):
                    full_response += response_chunk
                    message_placeholder.markdown(full_response + "▌")
                
                # Display final response
                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_message = f"I'm sorry, but I encountered an error while processing your question. Please try again."
                message_placeholder.markdown(error_message)
                st.error(f"Error details: {str(e)}")

    # Add disclaimer at the bottom
    st.markdown("""
    <div class="disclaimer">
        <p><strong>Medical Disclaimer:</strong> This chatbot provides general information and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
        <p>For in-person care, visit your nearest MyDoctor clinic.</p>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    
    if st.button("Retry"):
        st.experimental_rerun()