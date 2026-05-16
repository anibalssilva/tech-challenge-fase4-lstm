# Monitoramento

O requisito do Tech Challenge pede monitoramento da performance do modelo em produção, incluindo tempo de resposta e utilização de recursos.

Este projeto implementa um monitoramento simples diretamente na API.

## O que é monitorado

Endpoint:

```text
GET /metrics
```

Métricas:

- `uptime_seconds`: tempo de execução da API
- `request_count`: total de requisições
- `prediction_count`: total de chamadas de predição
- `error_count`: total de respostas com erro
- `avg_latency_ms`: tempo médio de resposta
- `last_latency_ms`: tempo da última requisição
- `cpu_percent`: uso de CPU do processo
- `memory_mb`: uso de memória do processo

## Por que isso é suficiente para o Tech Challenge

Para um projeto acadêmico, isso demonstra:

- rastreabilidade mínima
- acompanhamento de latência
- acompanhamento de recursos
- visibilidade operacional da API

## Evoluções possíveis

Em um cenário real, recomenda-se evoluir para:

- Prometheus
- Grafana
- OpenTelemetry
- logs estruturados
- armazenamento histórico das predições
- monitoramento de drift
- monitoramento de qualidade das previsões em produção
