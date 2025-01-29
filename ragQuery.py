import os
import logging
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class RAGQuery:
    def __init__(self):
        try:
            # Configuração do Qdrant
            self.qdrant_client = QdrantClient(
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                api_key=os.getenv("QDRANT_API_KEY")
            )
            
            # Configuração de embeddings com OpenAI
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )

            # Configuração do VectorStore com Qdrant
            self.vectorstore = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name="investigacao",
                embedding=self.embeddings
            )
            logger.info("Qdrant VectorStore configurado com sucesso.")

            # Configuração do modelo de linguagem
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            logger.info("Modelo ChatGPT configurado com sucesso.")

            # Definir Prompt Template
            self.prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=(
                    "Você é um assistente que responde perguntas com base no seguinte contexto:\n\n"
                    "{context}\n\n"
                    "Pergunta: {question}\n\n"
                    "Responda de forma clara e objetiva, incluindo sempre a referência (procedimento) e o número do(s) documento(s) que embasam a resposta."
                )
            )
            logger.info("Prompt Template configurado com sucesso.")

            # Configuração do limite de resultados do Qdrant
            self.top_k = int(os.getenv("QDRANT_TOP_K", 10))

            # Configuração da cadeia de consulta
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": self.top_k}
                ),
                return_source_documents=True,
                chain_type="stuff",
                chain_type_kwargs={"prompt": self.prompt_template}
            )
            logger.info("Cadeia de consulta RetrievalQA configurada com sucesso.")
        except Exception as e:
            logger.error("Erro durante a configuração do RAGQuery: %s", e)
            raise

    def gerar_embedding(self, texto):
        return self.embeddings.embed_query(texto)

    def consultar(self, pergunta, referencia):
        try:
            logger.info("Deverá responder a pergunta: %s", pergunta)
            logger.info("Consultando RAG para a referência: %s", referencia)

            # Adicionar filtro por referência no Qdrant
            resultados = self.vectorstore.similarity_search(
                query=pergunta,
                filter={
                    "must": [
                        {
                            "key": "metadata.referencia",
                            "match": {"value": referencia}
                        }
                    ]
                },
                k=self.top_k
            )

            if not resultados:
                logger.warning("Nenhum documento encontrado para referência: %s", referencia)
                return "Nenhum documento relevante encontrado para essa referência."

            contexto = "\n\n".join([
                f"Informação: {r.page_content}\nProcedimento: {r.metadata['referencia']}\nDocumento: {r.metadata['documento']}"
                for r in resultados
            ])

            resposta = self.qa_chain.invoke({"query": pergunta})

            logger.info("Consulta ao RAG concluída com sucesso para a referência: %s", referencia)
            return resposta["result"]
        except Exception as e:
            logger.error("Erro ao consultar o RAG para a referência %s: %s", referencia, e)
            return "Erro ao realizar a consulta. Verifique os logs para mais detalhes."

if __name__ == "__main__":
    query_processor = RAGQuery()

    resposta = query_processor.consultar("Qual é o valor do recibo do veículo.", "IP 1234/2025")
    print(resposta)