import os
import logging
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from weaviate import WeaviateClient, auth, connect
from langchain_community.vectorstores import Weaviate
from langchain_openai.embeddings import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class RAGQuery:
    def __init__(self):
        try:
            weaviate_client = WeaviateClient(
                connection_params=weaviate.connect.ConnectionParams.from_url(
                    url=os.getenv("WEAVIATE_HOST"),
                    auth_credentials=auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
                )
            )

            # Configuração de embeddings com OpenAI
            self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

            # Configuração do VectorStore com LangChain
            self.vectorstore = Weaviate(
                client=weaviate_client,
                index_name="Investigacao",
                embedding_function=self.embeddings
            )
            print("Weaviate VectorStore configurado com sucesso.")

            # Configuração do modelo de linguagem
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            logger.info("Modelo ChatGPT configurado com sucesso.")

            # Definir Prompt Template
            self.prompt_template = PromptTemplate(
                input_variables=["contexto", "pergunta"],
                template=(
                    "Você é um assistente que responde perguntas com base no seguinte contexto:\n\n"
                    "{contexto}\n\n"
                    "Pergunta: {pergunta}\n\n"
                    "Responda de forma clara e objetiva, incluindo sempre a referência (procedimento) e o documento original onde a informação foi encontrada."
                )
            )
            logger.info("Prompt Template configurado com sucesso.")

            # Configuração do limite de resultados do Weaviate
            self.top_k = int(os.getenv("WEAVIATE_TOP_K", 10))  # Padrão: 10 resultados

            # Configuração da cadeia de consulta
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": self.top_k}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt_template}
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

            # Adicionar filtro por referência no Weaviate
            resultados = self.vectorstore.similarity_search(
                query=pergunta,
                filter={
                    "path": ["referencia"],
                    "operator": "Equal",
                    "valueString": referencia
                },
                k=self.top_k
            )

            # Verificar se há documentos relevantes
            if not resultados:
                logger.warning("Nenhum documento encontrado para referência: %s", referencia)
                return "Nenhum documento relevante encontrado para essa referência."

            # Preparar o contexto incluindo referência e documento
            contexto = "\n\n".join([
                f"Informação: {r['text']}\nProcedimento: {r['referencia']}\nDocumento: {r['documento']}"
                for r in resultados
            ])

            # Gerar resposta usando o modelo de linguagem
            resposta = self.qa_chain.run({"contexto": contexto, "pergunta": pergunta})

            logger.info("Consulta ao RAG concluída com sucesso para a referência: %s", referencia)
            return resposta
        except Exception as e:
            logger.error("Erro ao consultar o RAG para a referência %s: %s", referencia, e)
            return "Erro ao realizar a consulta. Verifique os logs para mais detalhes."

if __name__ == "__main__":
    query_processor = RAGQuery()
