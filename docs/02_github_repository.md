# GitHub Repository

## Objetivo

O GitHub será o ponto central entre o desenvolvimento no Google Colab e o deploy no Render.

## Fluxo recomendado

```text
Colab -> commit/push -> GitHub -> Render
```

## Criar repositório

1. Acesse GitHub
2. Crie um novo repositório
3. Nome sugerido:

```text
tech-challenge-fase4-lstm-stock-api
```

4. Deixe como público ou privado
5. Faça upload dos arquivos deste projeto ou clone no Colab

## Branch

Use a branch `main`.

## Arquivos importantes para o Render

O Render precisa encontrar no repositório:

```text
Dockerfile
requirements.txt
app/
models/
render.yaml
```

## Artefatos do modelo

Para simplificar o Tech Challenge, este projeto espera que os arquivos do modelo treinado estejam no repositório:

```text
models/model.keras
models/scaler.pkl
models/metadata.json
```

Em projetos reais, uma alternativa melhor seria usar object storage, model registry ou Git LFS.

## Commit sugerido

```bash
git add .
git commit -m "Add LSTM stock prediction API"
git push origin main
```

Depois de treinar:

```bash
git add models reports
git commit -m "Add trained LSTM model artifacts"
git push origin main
```
