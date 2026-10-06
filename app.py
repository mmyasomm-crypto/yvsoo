import gradio as gr  
from app.main import app as fastapi_app  
app = gr.mount_gradio_app(fastapi_app, gr.Blocks(), path="/")  
