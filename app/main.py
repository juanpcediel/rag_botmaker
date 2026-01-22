from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import logging

from app.vectorstore import load_vectorstore
from app.rag import generate_answer
from app.llm_bedrock import call_llm
from app.memory_store import get_session_memory

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

store = load_vectorstore()
app = FastAPI()


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    memory = get_session_memory(req.session_id)
    answer, products = generate_answer(store, req.message, memory, call_llm)
    return {"answer": answer,
            "products": products,
            }


def gradio_chat(msg, history, session_id):
    history = history or []

    res = chat(ChatRequest(session_id=session_id, message=msg))
    answer = res["answer"]

    print(">>> ANSWER TO UI:", answer)  # DEBUG visible

    history.append({
        "role": "user",
        "content": msg
    })
    history.append({
        "role": "assistant",
        "content": answer
    })

    return "", history, session_id


with gr.Blocks() as demo:
    session_id = gr.State("local-session")

    chatbot = gr.Chatbot()
    txt = gr.Textbox()

    txt.submit(
        gradio_chat,
        [txt, chatbot, session_id],
        [txt, chatbot, session_id]
    )

app = gr.mount_gradio_app(app, demo, path="/gradio")
