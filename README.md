# Teoria das Filas — Calculadora (P108)

Este repositório contém uma aplicação Streamlit para exploração e cálculo de modelos clássicos de teoria das filas, desenvolvida como apoio à disciplina P108 — Otimização II (Inatel).

**Principais funcionalidades**
- Cálculo e visualização dos modelos: M/M/1, M/M/s, M/M/1/K, M/M/s/K, M/M/1/N, M/M/s/N, M/G/1.
- Análise de sistemas com prioridades (preemptivo e não-preemptivo).
- Exibição de métricas-chave (P0, ρ, L, Lq, W, Wq), distribuições P(n) e gráficos.
- Painel interativo com controles para parâmetros (λ, μ, s, K, N, σ², classes de prioridade).

**Arquivos principais**
- [filas_app.py](filas_app.py) — aplicação Streamlit principal com a lógica dos modelos e interface.

**Requisitos**
- Python 3.8+ recomendado
- Bibliotecas Python:
	- streamlit
	- pandas
	- matplotlib
	- numpy
	- (opcional) outras bibliotecas padrão como `math` e `traceback`

Você pode instalar as dependências com:

```bash
python -m pip install streamlit pandas matplotlib numpy
```

**Como executar**

No terminal, a partir da raiz do projeto execute:

```bash
streamlit run filas_app.py
```

Em seguida abra o link informado pelo Streamlit (normalmente http://localhost:8501).

**Visão rápida da interface**
- Barra lateral: escolha do modelo, parâmetros (λ, μ, s, K, N, σ²), unidades de tempo e classes de prioridade.
- Aba "Calculadora": execução do modelo selecionado, tabela P(n), métricas e gráficos.
- Aba "Teoria & Fórmulas": resumo das fórmulas usadas para cada modelo.
- Aba "Análise de Sensibilidade": variação de K/N para analisar comportamento de L, Lq, W, Wq e P0.

**Modelos implementados**
- M/M/1, M/M/s (Erlang-C), M/M/1/K, M/M/s/K, M/M/1/N, M/M/s/N, M/G/1 (Pollaczek–Khinchine), Prioridades (preemptivo e não-preemptivo).

**Exemplo rápido**
- Selecionar "M/M/1" na barra lateral, definir `λ=3` e `μ=4` e observar as métricas exibidas.

**Contribuições e próximos passos**
- Sugestões: adicionar `requirements.txt`, testes automatizados e exemplos predefinidos de cenários.
- Abra uma issue ou envie um pull request com melhorias.

---