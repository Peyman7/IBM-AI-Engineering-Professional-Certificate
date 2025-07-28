# Configuration file for all constants and parameters

MODEL_ID = "mistralai/mixtral-8x7b-instruct-v01"
EMBED_MODEL_ID = "ibm/slate-125m-english-rtrvr"
PROJECT_ID = "skills-network"
IBM_CLOUD_URL = "https://us-south.ml.cloud.ibm.com"

GEN_PARAMS = {
    "max_new_tokens": 256,
    "temperature": 0.5,
}

EMBED_PARAMS = {
    "truncate_input_tokens": 3,
    "return_options": {"input_text": True},
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
