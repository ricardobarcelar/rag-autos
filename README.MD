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


## 4. Qdrant - VectorStore

Porta de gerenciamento: 6333
Dashboard: http://localhost:6333
Outras documentações:
https://api.qdrant.tech/api-reference/collections/get-collection


Criar uma collection:
```
PUT collections/investigacao
{
  "vectors": {
    "size": 1536,
    "distance": "Cosine"
  }
}
```

Listar alguns itens:
```
POST /collections/investigacao/points/scroll
{
    "limit": 15,
    "with_vectors": true
}
```

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
curl -X POST http://localhost:8000/consultar \
-H "Content-Type: application/json" \
-d '{
  "referencia": "IP 1234/2025",
  "pergunta": "Qual é o resumo do caso?"
}'
```

#


##### 👤 Ricardo R Barcelar
GitHub: @ricardobarcelar

LinkedIn: linkedin.com/in/ricardo-barcelar-917abb332

Email: ricardobarcelar@gmail.com