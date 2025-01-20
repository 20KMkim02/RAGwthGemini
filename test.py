import wandb
import weave
import streamlit as st
import requests
from script import TFIDFRetriever

# WandB Initialization
WANDB_PROJECT = "RAG_Streamlit"
wandb.login()

run = wandb.init(
    project=WANDB_PROJECT,
    group="EiEi",
)
weave_client = weave.init(WANDB_PROJECT)

# Retrieve and index chunked data
chunked_data = weave.ref("chunked_data:v1").get()
retriever = TFIDFRetriever()
retriever.index_data(list(map(dict, chunked_data.rows)))

# Streamlit App Title
st.title("💬 Chat with ME eiei")

# Sidebar input for Gemini API key
gemini_api_key = st.sidebar.text_input("Enter your Gemini API key", type="password")

# Function to generate response from Gemini API
def generate_response(input_text):
    gemini_api_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"
    
    if not gemini_api_key:
        st.warning("Please provide a valid Gemini API key!", icon="⚠")
        return "Error: No API key"

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": input_text}]
        }]
    }

    try:
        response = requests.post(gemini_api_endpoint, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and result["candidates"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "No content returned from Gemini API."
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"An error occurred: {e}"

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Store chat history

# Chat Interface
st.write("### Chat History")

# Footer input box
st.write("---")  # Separator line
input_text = st.chat_input("Type your message here:")
if input_text:
    if not gemini_api_key:
        st.warning("Please provide a valid Gemini API key!")
    elif input_text.strip():
        # Add user message to session state
        st.session_state["messages"].append({"type": "user", "text": input_text})

        # Retrieve documents using TFIDFRetriever
        search_results = retriever.search(input_text)
        retrieved_content = "\n\n".join([result["text"] for result in search_results])
        
        # Combine retrieved documents with user input for context
        input_with_context = f"Query: {input_text}\n\nRetrieved Documents:\n{retrieved_content}"
        
        # Get response from Gemini API
        response = generate_response(input_with_context)
        
        # Add API response to session state
        st.session_state["messages"].append({"type": "bot", "text": response})
        
        wandb.log(input_text,retrieved_content,response)

# Display chat history
for message in st.session_state["messages"]:
    if message["type"] == "user":
        st.write(f"**You:** {message['text']}", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color: #d66e67; text-align: left;'><b>Gemini:</b> {message['text']}</div>", unsafe_allow_html=True)
