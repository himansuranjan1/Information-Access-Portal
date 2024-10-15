import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain
from PyPDF2.errors import PdfReadError  # Import the error for handling
from PyPDF2 import PdfReader

def user_input(user_question):
    # Check if conversation exists in session_state
    if 'conversation' in st.session_state and st.session_state['conversation'] is not None:
        conversation = st.session_state['conversation']
        
        # Get the response from the conversation chain
        response = conversation({'question': user_question})
        
        # Store chat history in session_state
        st.session_state['chathistory'] = response['chat_history']
        
        # Display chat history in the app
        for i, msg in enumerate(st.session_state['chathistory']):
            if i % 2 == 0:  # User inputs
                st.write("User:", msg.content)  # Access content attribute
            else:  # AI responses
                st.write("Reply:", msg.content)  # Access content attribute


def main():
    # Setting the page configuration for Streamlit
    st.set_page_config(page_title='INFORMATION-ACCESS', page_icon='🤩✨')
    
    st.header('INFORMATION-ACCESS Portal 🤩✨')
    
    # User input for asking questions
    user_question = st.text_input('Ask a question from the uploaded PDF files')

    # Initialize session state variables if not already present
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = None
    if 'chathistory' not in st.session_state:
        st.session_state['chathistory'] = []
    
    # If user provides input, process the question
    if user_question:
        user_input(user_question)

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.title('Menu')
        
        # File uploader for PDF files
        pdf_docs = st.file_uploader("Upload your PDF files and click on the submit button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner(text="Processing..."):
                if pdf_docs:
                    text = ""
                for pdf in pdf_docs:
                    try:
                        text += get_pdf_text([pdf])
                    except PdfReadError:
                        st.error(f"Error processing {pdf.name}: PDF might be corrupted or incomplete.")
                if text:
                    text_chunks = get_text_chunks(text)
                    vector_store = get_vector_store(text_chunks)
                    conversation = get_conversational_chain(vector_store)
                    st.session_state['conversation'] = conversation
                    st.success("Processing complete! You can now ask questions.")
                else:
                    st.error("Please upload at least one PDF file before submitting.")

    


    




if __name__ == '__main__':
    main()
