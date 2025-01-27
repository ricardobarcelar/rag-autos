# POC RAG Autos PJC

## 1. Tecnologias usadas

- Docker
- Python
- FastAPI
- Weaviate
- OpenAI API
- MinIO

## 2. Dependências do SO

```
tesseract-ocr tesseract-ocr-por ffmpeg
```

## 3. Processamento de áudio

https://alphacephei.com/vosk/models/vosk-model-pt-fb-v0.1.1-20220516_2113.zip


## 4. Weaviate - Vector Store

Porta de gerenciamento: 50051
Porta de serviço: 8080

### 4.1. Fornecer a Chave por meio de variáveis de ambiente

```
WEAVIATE_API_KEY=minha-chave-secreta
```

## 5. Variáveis de ambiente

Informadas por meio do arquivo [.env](.env) ou por meio de variáveis de ambiente na inicialização do container


## 6. Documentação da API

URL: http://localhost:8000/docs

## 7. Inciar o serviço

```
docker-compose up --build
```

## 8. Testar o serviço
```
curl -X POST http://localhost:8000/consulta \
-H "Content-Type: application/json" \
-d '{
  "referencia": "IP 1234/202",
  "pergunta": "Qual é o resumo do caso?"
}'
```

#


##### 👤 Ricardo R Barcelar
GitHub: @ricardobarcelar

LinkedIn: linkedin.com/in/ricardo-barcelar-917abb332

Email: ricardobarcelar@gmail.com