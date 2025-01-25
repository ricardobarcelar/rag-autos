from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ragQuery import RAGQuery
import os
import logging

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuração inicial da API
app = FastAPI(title="RAG API", description="API para consulta ao RAG", version="1.0.0")

# Modelo para entrada de dados
class ConsultaRequest(BaseModel):
    referencia: str
    pergunta: str

# Instância do RAGQuery
weaviate_url = os.getenv("WEAVIATE_HOST", "http://localhost:8080")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "")
openai_api_key = os.getenv("OPENAI_API_KEY", "")

try:
    rag_query = RAGQuery(weaviate_url, weaviate_api_key, openai_api_key)
    logger.info("Instância do RAGQuery criada com sucesso.")
except Exception as e:
    logger.error("Erro ao configurar RAGQuery: %s", e)
    raise RuntimeError(f"Erro ao configurar RAGQuery: {e}")

@app.post("/consultar", summary="Consulta ao RAG", response_description="Resposta gerada pelo RAG")
async def consultar_rag(request: ConsultaRequest):
    """
    Endpoint para consultar o RAG.

    :param request: JSON contendo a referência e a pergunta.
    :return: Resposta gerada pelo ChatGPT com base no conteúdo do RAG.
    """
    try:
        logger.info("Recebida consulta com referência: %s", request.referencia)
        resposta = rag_query.consultar(request.pergunta, request.referencia)
        logger.info("Consulta processada com sucesso para referência: %s", request.referencia)
        return {"referencia": request.referencia, "resposta": resposta}
    except Exception as e:
        logger.error("Erro ao consultar o RAG: %s", e)
        raise HTTPException(status_code=500, detail=f"Erro ao consultar o RAG: {str(e)}")

# Exemplo de inicialização da API
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
