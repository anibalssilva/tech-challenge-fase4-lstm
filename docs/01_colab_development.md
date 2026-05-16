# Desenvolvimento no Google Colab

Este projeto foi pensado para ser desenvolvido no Google Colab.

## Por quê

O Colab facilita o uso de GPU/CPU em nuvem, evita configuração pesada local e permite treinar o modelo sem depender do computador pessoal.

## Passo 1 — Criar ou clonar o repositório

No Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Depois clone seu repositório:

```bash
git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
cd SEU_REPOSITORIO
```

Se você ainda não criou o repositório, crie no GitHub e depois faça o clone.

## Passo 2 — Instalar dependências

```bash
pip install -r requirements.txt
```

## Passo 3 — Treinar o modelo

```bash
python -m src.training.train   --symbol DIS   --start-date 2018-01-01   --end-date 2024-07-20   --sequence-length 60   --epochs 40   --batch-size 32   --model-dir models   --report-dir reports
```

## Passo 4 — Conferir artefatos gerados

Depois do treino, confira se existem:

```text
models/model.keras
models/scaler.pkl
models/metadata.json
reports/metrics.json
reports/predictions.csv
reports/training_loss.png
```

## Passo 5 — Testar a API no Colab

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Em outra célula:

```bash
curl http://localhost:8000/health
```

## Passo 6 — Enviar para o GitHub

Configure suas variáveis no Colab:

```bash
export GIT_NAME="Seu Nome"
export GIT_EMAIL="seu-email@gmail.com"
export GITHUB_USERNAME="seu-usuario"
export GITHUB_REPO="seu-repositorio"
export GITHUB_TOKEN="cole-seu-token-aqui"
export GIT_BRANCH="main"
```

Depois:

```bash
bash scripts/git_push_from_colab.sh
```

## Cuidados

- Não coloque token do GitHub dentro do código.
- Não comite arquivos `.env`.
- Garanta que a pasta `models/` foi enviada para o GitHub antes do deploy no Render.
