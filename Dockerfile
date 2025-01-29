# Usar a imagem base do Python
FROM python:3.12.8-slim

# Definir o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copiar os arquivos do projeto para o contêiner
COPY . .

# Instalar dependências do sistema operacional
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-por \
    ffmpeg && \
    apt-get clean

# Instalar dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download pt_core_news_sm

# Expor a porta da aplicação
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["python", "consultaRequest.py"]
