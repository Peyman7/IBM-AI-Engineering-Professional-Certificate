import gradio as gr
from retriever_chain import run_qa

# Optional: Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def qa_wrapper(file, question):
    return run_qa(file.name, question)

interface = gr.Interface(
    fn=qa_wrapper,
    inputs=[
        gr.File(label="Upload PDF File", file_types=[".pdf"], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Response"),
    title="QA Chatbot",
    description="Upload a PDF and ask a question. The chatbot will try to answer using the provided document."
)

if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", server_port=7860)
