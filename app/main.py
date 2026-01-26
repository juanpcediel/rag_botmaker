from fastapi import FastAPI
from pydantic import BaseModel
import logging
import gradio as gr
import os

from app.vectorstore import load_vectorstore
from app.rag import generate_answer
from app.llm_bedrock import call_llm
from app.memory_store import get_session_memory


# Log for debuging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger('rag-api')

# Core app RAG
store = load_vectorstore()
app = FastAPI()

# Models 
class ChatRequest(BaseModel):
    session_id: str
    message: str

# Endpoint fastAPI
@app.post("/chat")
def chat(req: ChatRequest):
    
    logger.info(f'Chat Request | Session = {req.session_id}')

    memory = get_session_memory(req.session_id)
    answer, products = generate_answer(
        store, 
        req.message, 
        memory, 
        call_llm)
    
    return {"answer": answer,
            "products": products,
            }


def gradio_chat(message, history, session_id):
    history = history or []

    response = chat(ChatRequest(session_id=session_id, message=message))
    answer = response["answer"]

    print(">>> ANSWER TO UI:", answer)  # DEBUG visible en gradio

    history.append({
        "role": "user",
        "content": message
    })
    history.append({
        "role": "assistant",
        "content": answer
    })

    return "", history, session_id

def build_gradio():
    with gr.Blocks(title = 'RAG QA Console') as demo:
        session_id = gr.State("local_qa-session")

        gr.Markdown('## RAG-QA Interno')
        
        chatbot = gr.Chatbot()
        txt = gr.Textbox(placeholder='Escribe tu consulta: ')

        txt.submit(
            gradio_chat,
            [txt, chatbot, session_id],
            [txt, chatbot, session_id]
        )
    return demo

# app = gr.mount_gradio_app(app, demo, path="/gradio")

# Montado explicito y aislado
ENABLE_QA = os.getenv('ENABLE_QA', 'false').lower() == 'true'
if ENABLE_QA:
        
    app = gr.mount_gradio_app(
        app,
        build_gradio(),
        path='/qa'
    )