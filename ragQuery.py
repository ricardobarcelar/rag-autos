import logging
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Weaviate

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGQuery:
    def __init__(self, weaviate_url, weaviate_api_key, openai_api_key):
        try:
            # Configuração da conexão com Weaviate
            self.vectorstore = Weaviate(
                weaviate_url,
                "Investigacao",
                api_key=weaviate_api_key
            )
            logger.info("Conexão com Weaviate configurada com sucesso.")

            # Configuração do modelo de linguagem (ChatGPT via OpenAI)
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0,
                openai_api_key=openai_api_key
            )
            logger.info("Modelo ChatGPT configurado com sucesso.")

            # Configuração da cadeia de recuperação e resposta
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever()
            )
            logger.info("Cadeia de consulta RetrievalQA configurada com sucesso.")
        except Exception as e:
            logger.error("Erro durante a configuração do RAGQuery: %s", e)
            raise

    def consultar(self, pergunta, referencia):
        """
        Consulta o RAG usando LangChain e retorna uma resposta.

        :param pergunta: A pergunta feita pelo usuário.
        :param referencia: O auto de investigação para filtrar documentos.
        :return: Resposta gerada pelo ChatGPT com base nos documentos recuperados.
        """
        try:
            logger.info("Consultando RAG para a referência: %s", referencia)

            # Adicionar filtro por referência nos metadados
            self.vectorstore.add_filter(lambda doc: doc["referencia"] == referencia)

            # Realizar a consulta
            resposta = self.qa_chain.run(pergunta)

            logger.info("Consulta ao RAG concluída com sucesso para a referência: %s", referencia)
            return resposta
        except Exception as e:
            logger.error("Erro ao consultar o RAG para a referência %s: %s", referencia, e)
            return "Erro ao realizar a consulta. Verifique os logs para mais detalhes."

# Exemplo de uso
if __name__ == "__main__":
    weaviate_url = "http://localhost:8080"
    weaviate_api_key = "sua_api_key"
    openai_api_key = "sua_openai_api_key"

    try:
        rag_query = RAGQuery(weaviate_url, weaviate_api_key, openai_api_key)

        pergunta = "Quais são os detalhes do depoimento sobre o evento X?"
        referencia = "IP 10/2024"

        resposta = rag_query.consultar(pergunta, referencia)
        print("Resposta:", resposta)
    except Exception as e:
        logger.error("Erro durante a execução do exemplo de uso: %s", e)
