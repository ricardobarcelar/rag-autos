import os
import json
import logging
import psycopg2
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from minio import Minio
from langchain.vectorstores import Weaviate
from langchain.embeddings.openai import OpenAIEmbeddings
from binaryProcessor import PDFProcessor, ImageProcessor, AudioProcessor, VideoProcessor, FormatSupport

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FilaProcessor:
    def __init__(self):
        try:
            # Configuração do PostgreSQL
            self.pg_conn = psycopg2.connect(
                host=os.getenv("PG_HOST"),
                port=os.getenv("PG_PORT"),
                dbname=os.getenv("PG_DATABASE"),
                user=os.getenv("PG_USER"),
                password=os.getenv("PG_PASSWORD")
            )
            self.pg_cursor = self.pg_conn.cursor()
            logger.info("Conexão com PostgreSQL estabelecida com sucesso.")
        except Exception as e:
            logger.error("Erro ao conectar ao PostgreSQL: %s", e)
            raise

        try:
            # Configuração do MinIO
            self.minio_client = Minio(
                os.getenv("MINIO_HOST"),
                access_key=os.getenv("MINIO_USER"),
                secret_key=os.getenv("MINIO_PASSWORD"),
                secure=False
            )
            logger.info("Conexão com MinIO estabelecida com sucesso.")
        except Exception as e:
            logger.error("Erro ao conectar ao MinIO: %s", e)
            raise

        try:
            # Configuração do LangChain com Weaviate
            self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            self.vectorstore = Weaviate(
                url=os.getenv("WEAVIATE_HOST"),
                index_name="Investigacao",
                api_key=os.getenv("WEAVIATE_API_KEY"),
                embedding_function=self.embeddings
            )
            logger.info("Conexão com Weaviate configurada com sucesso.")
        except Exception as e:
            logger.error("Erro ao configurar Weaviate: %s", e)
            raise

        # Processadores especializados
        self.pdf_processor = PDFProcessor()
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()

        # Agendador
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self.processar_fila, 'interval', minutes=5)
        self.scheduler.start()
        logger.info("Agendador configurado e iniciado.")

    def processar_fila(self):
        logger.info("Iniciando processamento da fila...")

        try:
            # Buscar itens pendentes
            self.pg_cursor.execute("""
                SELECT id, referencia, tipo, acao, conteudo
                FROM fila_rag
                WHERE data_hora_processamento IS NULL
                ORDER BY data_hora ASC
                LIMIT 10;
            """)
            itens = self.pg_cursor.fetchall()
        except Exception as e:
            logger.error("Erro ao buscar itens da fila: %s", e)
            return

        for item in itens:
            id_fila, referencia, tipo, acao, conteudo = item
            logger.info("Processando item %s, referência: %s, ação: %s", id_fila, referencia, acao)

            try:
                if acao == 'I':  # Incluir
                    if tipo == 'E':  # Dados estruturados
                        self.processar_estruturado(id_fila, referencia, conteudo)
                    elif tipo == 'B':  # Binários
                        self.processar_binario(id_fila, conteudo)
                elif acao == 'E':  # Excluir
                    self.remover_do_rag(id_fila)

                # Atualiza status no PostgreSQL
                self.pg_cursor.execute("""
                    UPDATE fila_rag
                    SET data_hora_processamento = %s
                    WHERE id = %s;
                """, (datetime.now(), id_fila))
                self.pg_conn.commit()
                logger.info("Item %s processado com sucesso.", id_fila)
            except Exception as e:
                logger.error("Erro ao processar item %s: %s", id_fila, e)

        logger.info("Processamento da fila concluído.")

    def processar_estruturado(self, id_fila, referencia, conteudo):
        try:
            logger.info("Processando dados estruturados: %s", referencia)
            self.vectorstore.add_texts(
                texts=[conteudo],
                metadatas=[{"id_fila": id_fila, "referencia": referencia}]
            )
            logger.info("Dados estruturados armazenados no Weaviate para ID %s", id_fila)
        except Exception as e:
            logger.error("Erro ao processar dados estruturados %s: %s", referencia, e)

    def processar_binario(self, id_fila, conteudo):
        try:
            data = json.loads(conteudo)
            bucket, file_hash = data["bucket"], data["hash"]
            logger.info("Baixando binário %s do bucket %s", file_hash, bucket)

            # Download do arquivo
            arquivo_destino = f"/tmp/{file_hash}"
            self.minio_client.fget_object(bucket, file_hash, arquivo_destino)

            # Identificar tipo e processar
            tipo_arquivo = FormatSupport.verificar_formato(arquivo_destino)
            if tipo_arquivo == "pdf":
                texto = self.pdf_processor.processar_pdf(arquivo_destino)
            elif tipo_arquivo == "imagem":
                texto = self.image_processor.processar_imagem(arquivo_destino)
            elif tipo_arquivo == "audio":
                texto = self.audio_processor.processar_audio(arquivo_destino)
            elif tipo_arquivo == "video":
                texto = self.video_processor.processar_video(arquivo_destino)
            else:
                logger.warning("Tipo de arquivo não suportado: %s", file_hash)
                return

            self.vectorstore.add_texts(
                texts=[texto],
                metadatas=[{"id_fila": id_fila, "referencia": ""}]
            )
            logger.info("Binário processado e armazenado no Weaviate para ID %s", id_fila)
        except Exception as e:
            logger.error("Erro ao processar binário %s: %s", id_fila, e)

    def remover_do_rag(self, id_fila):
        try:
            logger.info("Removendo ID %s do RAG", id_fila)
            self.vectorstore.delete(ids=[str(id_fila)])
            logger.info("ID %s removido do Weaviate", id_fila)
        except Exception as e:
            logger.error("Erro ao remover ID %s do RAG: %s", id_fila, e)

if __name__ == "__main__":
    processor = FilaProcessor()
