import os
import streamlit as st
from dotenv import load_dotenv
import time
import streamlit.components.v1 as components

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
    
    /* Patient data form styling */
    .patient-form {
        background-color: #f5f9fa;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    
    .form-header {
        color: #087484;
        font-size: 1.2rem;
        margin-bottom: 15px;
        border-bottom: 1px solid #ddd;
        padding-bottom: 5px;
    }
    
    .toggle-container {
        background-color: #f5f9fa;
        padding: 10px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid #e0e0e0;
    }
    /* Style for clear chat button */
    .clear-button {
        background-color: #f0f0f0;
        color: #333;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-size: 0.8rem;
        cursor: pointer;
        transition: all 0.3s;
        float: right;
        margin-bottom: 10px;
    }
    
    .clear-button:hover {
        background-color: #ff9414;
        color: white;
        border-color: #ff9414;
    }
    
    .button-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "use_patient_data" not in st.session_state:
    st.session_state.use_patient_data = False

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {
        "patient_id": "",
        "name": "",
        "age": "",
        "location": "",
        "budget": "Mid-Low",
        "critical_conditions": [],
        "past_medical_history": [],
        "current_medications": [],
        "allergies": [],
        "vaccination_history": []
    }

if "patient_data_submitted" not in st.session_state:
    st.session_state.patient_data_submitted = False

# Logo and App title
st.markdown('<div class="logo-container"><div class="logo">MD</div></div>', unsafe_allow_html=True)
st.markdown("<h1 class='main-header'>MyDoctor Health Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Your personal health guide</p>", unsafe_allow_html=True)
def clear_chat_history():
    """Clear the chat history in session state"""
    if "messages" in st.session_state:
        st.session_state.messages = []
    st.rerun()
# Toggle for patient data usage
with st.container():
    st.markdown('<div class="toggle-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**Use Patient Information**")
        st.caption("Enable to provide personalized health advice based on your medical history")
    with col2:
        use_patient_data = st.toggle(
    "Use Patient Information", 
    value=st.session_state.use_patient_data, 
    key="toggle_patient_data",
    label_visibility="collapsed" 
)
        if use_patient_data != st.session_state.use_patient_data:
            st.session_state.use_patient_data = use_patient_data
            st.session_state.patient_data_submitted = False
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Patient data form
if st.session_state.use_patient_data and not st.session_state.patient_data_submitted:
    with st.container():
        st.markdown('<div class="patient-form">', unsafe_allow_html=True)
        st.markdown('<div class="form-header">Patient Information</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic information
            st.session_state.patient_data["name"] = st.text_input("Full Name", 
                                                                 value=st.session_state.patient_data["name"])
            st.session_state.patient_data["patient_id"] = st.text_input("Patient ID (Optional)", 
                                                                        value=st.session_state.patient_data["patient_id"])
            st.session_state.patient_data["age"] = st.number_input("Age", min_value=1, max_value=120, 
                                                                 value=int(st.session_state.patient_data["age"]) if st.session_state.patient_data["age"] else 35)
            
            # Location and budget
            locations = ["Kampala", "Entebbe", "Jinja", "Gulu", "Mbarara", "Other"]
            location_index = locations.index(st.session_state.patient_data["location"]) if st.session_state.patient_data["location"] in locations else 0
            st.session_state.patient_data["location"] = st.selectbox("Location", options=locations, index=location_index)
            
            budget_options = ["Low", "Mid-Low", "Mid", "Mid-High", "High"]
            budget_index = budget_options.index(st.session_state.patient_data["budget"]) if st.session_state.patient_data["budget"] in budget_options else 1
            st.session_state.patient_data["budget"] = st.selectbox("Budget Level", options=budget_options, index=budget_index)
        
        with col2:
            # Medical information
            critical_conditions_options = ["diabetic", "hypertensive", "pregnant", "asthmatic", "HIV+", "cardiac condition"]
            st.session_state.patient_data["critical_conditions"] = st.multiselect(
                "Critical Conditions", 
                options=critical_conditions_options,
                default=st.session_state.patient_data["critical_conditions"]
            )
            
            # Common medications
            common_medications = ["Metformin", "Insulin", "Paracetamol", "Ibuprofen", "Hydrochlorothiazide", "Enalapril"]
            st.session_state.patient_data["current_medications"] = st.multiselect(
                "Current Medications",
                options=common_medications,
                default=st.session_state.patient_data["current_medications"]
            )
            
            # Common allergies
            common_allergies = ["Penicillin", "Sulfa", "Latex", "Nuts", "Pollen", "None"]
            st.session_state.patient_data["allergies"] = st.multiselect(
                "Allergies",
                options=common_allergies,
                default=st.session_state.patient_data["allergies"]
            )
            
            # Vaccinations
            common_vaccinations = ["COVID-19", "Hepatitis B", "Tetanus", "Typhoid", "Yellow Fever"]
            st.session_state.patient_data["vaccination_history"] = st.multiselect(
                "Vaccination History",
                options=common_vaccinations,
                default=st.session_state.patient_data["vaccination_history"]
            )
            
            # Additional medical history
            common_history = ["Surgery", "Malaria", "Typhoid", "Tuberculosis", "Hepatitis"]
            st.session_state.patient_data["past_medical_history"] = st.multiselect(
                "Past Medical History",
                options=common_history,
                default=st.session_state.patient_data["past_medical_history"]
            )
        
        if st.button("Save and Start Chat", type="primary"):
            if not st.session_state.patient_data["name"]:
                st.error("Please enter your name to continue")
            else:
                st.session_state.patient_data_submitted = True
                st.success("Patient information saved successfully!")
                st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)

# Emergency warning
st.markdown("""
<div class="emergency-warning">
    ⚠️ If you are experiencing a medical emergency, please call emergency services immediately or go to your nearest emergency department.
</div>
""", unsafe_allow_html=True)
if st.session_state.messages:  # Only show if there are messages
    cols = st.columns([4, 1])
    with cols[1]:
        if st.button("Clear Chat", type="secondary", use_container_width=True):
            clear_chat_history()
# Check if patient data is required but not submitted
if st.session_state.use_patient_data and not st.session_state.patient_data_submitted:
    st.info("Please fill in your patient information to start the chat")
else:
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
            patient_name = ""
            if st.session_state.use_patient_data and st.session_state.patient_data_submitted:
                patient_name = st.session_state.patient_data.get("name", "").split()[0]
                greeting = f"Hi {patient_name}! I'm MyDoctor's digital health assistant."
            else:
                greeting = "Hi there! I'm MyDoctor's digital health assistant."
                
            welcome_message = f"{greeting} You can tell me how you're feeling, and I'll try to help. Tell me about the symptoms you're experiencing!"
            st.session_state.messages.append(
                {"role": "assistant", "content": welcome_message}
            )

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

       # Chat input section with fixed streaming implementation
        if prompt := st.chat_input("Type your symptoms or health question..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message in the chat UI
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                try:
                    # Create a proper conversation history from session state
                    conversation_history = list(st.session_state.messages)
                    
                    # Get patient data if enabled
                    patient_id = None
                    if st.session_state.use_patient_data and st.session_state.patient_data_submitted:
                        # Use UUID as patient ID if not provided
                        patient_id = st.session_state.patient_data.get("patient_id", "") or f"UG{int(time.time())}"
                        # Set the patient data in the RAG
                        rag.set_patient_data(patient_id, st.session_state.patient_data)
                    
                    # Show typing indicator
                    message_placeholder = st.empty()
                    message_placeholder.markdown("""
                        <div style="display: flex; align-items: center; margin: 0;">
                            <div style="width: 16px; height: 16px; border: 2px solid #f3f3f3; 
                                    border-top: 2px solid #087484; border-radius: 50%; 
                                    animation: spin 1s linear infinite; margin-right: 8px;"></div>
                            <div style="color: #087484; font-style: italic; font-size: 15px;">Bot is typing...</div>
                        </div>
                        <style>
                            @keyframes spin {
                                0% { transform: rotate(0deg); }
                                100% { transform: rotate(360deg); }
                            }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    # Get the complete response
                    response_json = next(rag.generate_answer_stream(
                        query=prompt, 
                        patient_id=patient_id,
                        message_history=conversation_history
                    ))
                    
                    import json
                    import time
                    
                    response_data = json.loads(response_json)
                    response_text = response_data.get("response", "")
                    is_emergency = response_data.get("is_emergency", False)
                    
                    # Clear the typing indicator
                    message_placeholder.empty()
                    
                    # Create a custom text generator function with visible delay
                    def character_stream(text, delay=0.01):
                        """Generator that yields characters with a delay for visible streaming effect"""
                        accumulated_text = ""
                        for char in text:
                            accumulated_text += char
                            yield accumulated_text
                            time.sleep(delay)  # Add a small delay between characters
                    
                    # Two different approaches depending on if it's an emergency
                    if is_emergency:
                        # For emergency messages, create a styled emergency message
                        emergency_placeholder = st.empty()
                        
                        # Stream the text and update the emergency container with each iteration
                        for partial_text in character_stream(response_text):
                            emergency_placeholder.markdown(f"""
                            <div style="background-color: #FFEBEE; border-left: 5px solid #C62828; 
                                    color: #C62828; padding: 15px; border-radius: 5px; 
                                    margin: 10px 0; font-weight: bold; animation: pulse 2s infinite;">
                                <div style="font-size: 1.5rem; margin-bottom: 8px;">⚠️ URGENT MEDICAL ATTENTION NEEDED</div>
                                {partial_text}
                            </div>
                            <style>
                            @keyframes pulse {{
                                0% {{ box-shadow: 0 0 0 0 rgba(198, 40, 40, 0.4); }}
                                70% {{ box-shadow: 0 0 0 10px rgba(198, 40, 40, 0); }}
                                100% {{ box-shadow: 0 0 0 0 rgba(198, 40, 40, 0); }}
                            }}
                            </style>
                            """, unsafe_allow_html=True)
                        
                        # Log the emergency
                        import logging
                        logging.warning(f"EMERGENCY detected in query: '{prompt}'")
                    else:
                        # For normal responses, create a placeholder and update it with each iteration
                        response_placeholder = st.empty()
                        
                        # Stream the text with a visible typing effect
                        for partial_text in character_stream(response_text):
                            response_placeholder.markdown(partial_text)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "is_emergency": is_emergency
                    })
                    
                except Exception as e:
                    error_message = "I'm sorry, but I encountered an error while processing your question. Please try again."
                    st.error(f"Error details: {str(e)}")
                    import traceback
                    print(f"Error details: {traceback.format_exc()}")
                    
                    # Add error response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "is_emergency": False
                    })
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
            st.rerun()