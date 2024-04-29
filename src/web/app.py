import streamlit as st
from model_gen import load_model_and_tokenizer, gen

model, tokenizer = load_model_and_tokenizer()

st.set_page_config(page_title="Sabia-7b Instruct",
                   page_icon="🐦")

st.title('Sabia7B Instruct 🐦')

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Como posso te ajudar?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Diga algo"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = gen(prompt, model, tokenizer)
            st.markdown(response)
    
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)