from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from config import *

def get_llm():
    return WatsonxLLM(
        model_id=MODEL_ID,
        url=IBM_CLOUD_URL,
        project_id=PROJECT_ID,
        params={
            GenParams.MAX_NEW_TOKENS: GEN_PARAMS["max_new_tokens"],
            GenParams.TEMPERATURE: GEN_PARAMS["temperature"],
        }
    )

def get_embedding_model():
    return WatsonxEmbeddings(
        model_id=EMBED_MODEL_ID,
        url=IBM_CLOUD_URL,
        project_id=PROJECT_ID,
        params={
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: EMBED_PARAMS["truncate_input_tokens"],
            EmbedTextParamsMetaNames.RETURN_OPTIONS: EMBED_PARAMS["return_options"],
        }
    )
