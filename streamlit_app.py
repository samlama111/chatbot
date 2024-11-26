import streamlit as st
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Replace OpenAI API key section with model loading
@st.cache_resource  # This caches the model so it only loads once
def load_model():
    model_name_or_path = "samlama111/lora_model"  # Your model path
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_4bit=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer

# Load model and tokenizer
try:
    model, tokenizer = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using the local model
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_length=200,  # Adjust as needed
                num_return_sequences=1,
                temperature=0.7,  # Adjust for creativity vs consistency
                do_sample=True,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
