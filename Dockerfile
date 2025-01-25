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

# Baixar o modelo Vosk para reconhecimento de fala
RUN mkdir -p /app/vosk_models && \
    #wget -qO /app/vosk_models/vosk-model-small-pt-0.3.zip https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip && \
    unzip /app/vosk-model-small-pt-0.3.zip -d /app/vosk_models/ && \
    rm /app/vosk-model-small-pt-0.3.zip

# Instalar dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta da aplicação
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["python", "consultaRequest.py"]
