# Watsonx-Powered PDF QA Chatbot

This project is a **LangChain + IBM Watsonx + Gradio** application that lets users upload a PDF document and ask questions about its contents. It retrieves relevant chunks using embeddings and answers using a Watsonx LLM.

---

##  Features

- PDF upload and question answering
- IBM Watsonx foundation models
- IBM Watsonx embeddings
- LangChain-powered retrieval pipeline
- Gradio-based user interface
- Dockerized for easy deployment

---

##  Requirements

Before running locally or building Docker, ensure you have:

- Python 3.10+
- IBM Watsonx credentials and access
- Docker (optional for containerized run)

---

##  Setup (Local)

1. **Clone this repository**:

```bash
git clone https://github.com/your-username/watsonx-pdf-qa.git
cd watsonx-pdf-qa
