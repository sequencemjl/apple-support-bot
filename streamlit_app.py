import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

# --- PAGE SETUP ---
st.set_page_config(page_title="Apple Support AI", page_icon="🍎")
st.title("🍎 Apple Support AI Assistant")

# --- CONNECT TO DATABASE ---
# We try to get keys from the Cloud Config (Secrets)
try:
    # These secrets will be set in the Streamlit Dashboard later
    client = OpenAI(api_key=st.secrets["OPENAI_KEY"])
    pc = Pinecone(api_key=st.secrets["PINECONE_KEY"])
    index = pc.Index("apple-support")
except Exception:
    st.error("🚨 API Keys missing! Please set them in Streamlit Secrets.")
    st.stop()

# --- THE CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you today?"):
    # 1. Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Searching knowledge base..."):
        try:
            # 2. Convert text to numbers
            query_vector = client.embeddings.create(
                input=[prompt], 
                model="text-embedding-3-small"
            ).data[0].embedding

            # 3. Search Pinecone
            search_response = index.query(
                vector=query_vector,
                top_k=1,
                include_metadata=True
            )

            if search_response['matches']:
                best_match = search_response['matches'][0]
                past_q = best_match['metadata']['question']
                past_a = best_match['metadata']['answer']
                score = best_match['score']

                # 4. Generate Answer
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful Apple Support agent. Answer the user based on the context provided."},
                        {"role": "user", "content": f"User: {prompt}\n\nContext:\nQ: {past_q}\nA: {past_a}"}
                    ]
                )
                response_text = completion.choices[0].message.content
                
                # Append Reference for transparency
                full_response = f"{response_text}\n\n---\n*💡 Based on similar ticket (Match: {int(score*100)}%):*\n*Q: {past_q}*"

            else:
                full_response = "I couldn't find any similar past cases in our database."

            # 5. Show AI response
            st.chat_message("assistant").markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {e}")