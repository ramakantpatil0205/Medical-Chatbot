import streamlit as st
from backend.chat_backend import RAGChat
import base64, os, pathlib, textwrap

st.set_page_config(page_title='MediChat — Smart Medical Assistant', page_icon=':pill:', layout='wide')

# Custom CSS for nicer look
def local_css(css_txt):
    st.markdown(f"<style>{css_txt}</style>", unsafe_allow_html=True)

css = '''
.header {display:flex; align-items:center; gap:16px;}
.brand {font-size:28px; font-weight:700; color:#0b5cff;}
.subtitle {color:#6b7280;}
.chat-box {background:linear-gradient(180deg,#ffffff,#f7fbff); padding:18px; border-radius:12px; box-shadow:0 6px 24px rgba(11,92,255,0.06);}
.msg-user {background:#0b5cff;color:white;padding:10px;border-radius:10px;display:inline-block;}
.msg-bot {background:#eef2ff;color:#03132b;padding:10px;border-radius:10px;display:inline-block;}
'''
local_css(css)

# Header
st.markdown("""
<div class="header">
  <div>
    <div class="brand">MediChat</div>
    <div class="subtitle">Smart, explainable medical assistant — demo</div>
  </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])
with col1:
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    user_input = st.text_input('Ask a medical question (for demo use non-urgent questions):', 'What are symptoms of diabetes?')
    submitted = st.button('Ask MediChat')
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('### ⚙️ Settings')
    topk = st.slider('Top-k retrieval', 1, 5, 3)
    st.markdown('---')
    st.markdown('### 🔎 Sources')
    st.write('The app returns retrieved FAQ entries as sources for transparency.')

# Load/chat
try:
    chat = RAGChat(index_dir='indexes')
except Exception as e:
    st.error('Index not found. Run `python backend/build_faq_index.py --input sample_data/faqs.csv --output indexes` first.')
    st.stop()

if submitted and user_input.strip():
    with st.spinner('Thinking...'):
        answer, retrieved = chat.answer(user_input, topk=topk)
    # Display chat-like bubbles
    st.mark_markdown = st.markdown  # alias for type check safety
    st.markdown('<div style="margin-top:12px">', unsafe_allow_html=True)
    st.markdown(f'<div class="msg-user">You: {user_input}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="msg-bot"><b>MediChat:</b><br/>{answer.replace("\n","<br/>")}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # show retrieved sources
    st.markdown('### Retrieved sources & matches')
    for r in retrieved:
        st.write(f"- **Q:** {r['question']}  — _score: {r['score']:.4f}_")
        st.write(f"  - A: {r['answer']}")


# Footer
st.markdown('---')
st.markdown('Made with ❤️ for interview demos. Not medical advice. See LICENSE & MODEL_CARD for limitations.')
