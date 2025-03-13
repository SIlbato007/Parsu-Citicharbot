import streamlit as st
from rag_config import PSUChatBackend

# App configuration
st.set_page_config(
    page_title="PSU Chatbot",
    page_icon="🎓",
    layout="wide",
) 

# Initialize backend if not already in session state
if 'backend' not in st.session_state:
    st.session_state.backend = PSUChatBackend()
    # Automatically initialize the system
    with st.spinner("Initializing system. Please Wait..."):
        success, message = st.session_state.backend.initialize_system()
        if success:
            st.success(message)
        else:
            st.error(message)

# Initialize chat history in session state if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Application UI
st.title("ParSU Citicharbot")
st.write("Ask questions and information about Partido State university services and transactions.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
query = st.chat_input("Ask a question about Partido State University")

# Process user query
if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.write(query)
    
    # Generate response if backend is initialized
    if st.session_state.backend.chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                success, response = st.session_state.backend.generate_response(query)
                if success:
                    st.markdown(response, unsafe_allow_html=True)
                else:
                    st.error(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            message = "The system initialization failed. Please reload try to reload the app again"
            st.write(message)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": message})

# Add a footer
st.divider()
st.caption("Partido State University Chatbot - Powered by LLAMA")