## 1. Instalação das dependências

### 1.1. Dependências do SO

```
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-por
sudo apt install ffmpeg
```

### 1.2. Dependências do Python

[Requisitos](requirements.txt)

### 1.3. Para processamento de áudio usamos o Vosk

```
https://alphacephei.com/vosk/models/vosk-model-pt-fb-v0.1.1-20220516_2113.zip

```

## 2. Weaviate

Porta de gerenciamento: 50051
Porta de serviço: 8080

### 2.1. Criar uma chave de API

```
curl -X POST http://localhost:8080/v1/.well-known/apikeys \
     -H "Content-Type: application/json" \
     -d '{"description": "Chave para o sistema RAG", "key": "minha-chave-secreta"}'

```

Fornecer a Chave por meio de variáveis de ambiente

```
WEAVIATE_API_KEY=minha-chave-secreta
```


## 3. Variáveis de ambiente

Realizar por meio do arquivo .env

## 4. Executando o servidor

```
python rag_api.py
```

## 5. Documentação da API

URL: http://localhost:8000/docs

## 6. Docker-compose

```
docker-compose up --build

```