# Deploy no Render

Este projeto está pronto para deploy como **Web Service** no Render usando Docker.

## Pré-requisitos

- Repositório no GitHub com este projeto
- Arquivos do modelo dentro da pasta `models/`
- Conta no Render

## Opção A — Deploy via Dashboard

1. Acesse o Render
2. Clique em **New**
3. Selecione **Web Service**
4. Conecte seu repositório GitHub
5. Escolha o repositório do projeto
6. Configure:
   - Runtime: Docker
   - Dockerfile path: `./Dockerfile`
   - Branch: `main`
7. Crie as variáveis de ambiente:
   - `MODEL_DIR=/app/models`
   - `DEFAULT_SEQUENCE_LENGTH=60`
   - `LOG_LEVEL=INFO`
8. Clique em Deploy

## Opção B — Deploy via Blueprint

O projeto já possui `render.yaml`.

1. No Render, escolha Blueprint
2. Conecte o repositório
3. O Render lerá o arquivo `render.yaml`
4. Confirme a criação do serviço

## Health check

Depois do deploy, teste:

```bash
curl https://SUA-API.onrender.com/health
```

## Documentação Swagger

Acesse:

```text
https://SUA-API.onrender.com/docs
```

## Teste de previsão

```bash
curl -X POST "https://SUA-API.onrender.com/predict"   -H "Content-Type: application/json"   -d @examples/predict_request.json
```

## Possíveis problemas

### 1. API sobe, mas `/predict` retorna erro 503

Causa provável: arquivos do modelo não foram enviados para o GitHub.

Verifique se existem:

```text
models/model.keras
models/scaler.pkl
models/metadata.json
```

### 2. Build demora muito

TensorFlow é uma dependência pesada. Isso é esperado.

### 3. Serviço gratuito dorme

No plano gratuito, o serviço pode ficar inativo após um período sem uso. A primeira chamada pode demorar mais.

### 4. Erro de memória

Reduza o tamanho do modelo:

```bash
python -m src.training.train --lstm-units 32 --epochs 20
```
