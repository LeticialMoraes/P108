# P108 - Otimização 2

## Teoria das Filas 

Este repositório contém uma aplicação Streamlit para exploração e cálculo de modelos clássicos de teoria das filas, desenvolvida como apoio à disciplina P108 — Otimização II (Inatel).

**Principais funcionalidades**
- Cálculo e visualização dos modelos: M/M/1, M/M/s, M/M/1/K, M/M/s/K, M/M/1/N, M/M/s/N, M/G/1.
- Análise de sistemas com prioridades (preemptivo e não-preemptivo).
- Exibição de métricas-chave (P0, ρ, L, Lq, W, Wq), distribuições P(n) e gráficos.
- Painel interativo com controles para parâmetros (λ, μ, s, K, N, σ², classes de prioridade).

**Arquivos principais**
- `src/app.py` — aplicação Streamlit principal.
- `src/core.py` — implementações dos modelos de filas.
- `src/ui.py` — componentes de exibição e gráficos.
- `main.py` — wrapper de execução.
- `filas_app.py` — wrapper legado compatível com o nome original.

**Estrutura do projeto**
- `src/`
  - `app.py`
  - `core.py`
  - `ui.py`
- `main.py`
- `filas_app.py`
- `requirements.txt`
- `README.md`

**Requisitos**
- Python 3.8+ recomendado
- Bibliotecas Python:
	- streamlit
	- pandas
	- matplotlib
	- numpy
	- (opcional) outras bibliotecas padrão como `math` e `traceback`
- Arquivo de dependências: `requirements.txt`

Você pode instalar as dependências com:

```bash
python -m pip install streamlit pandas matplotlib numpy
```

**Como executar**

No terminal, a partir da raiz do projeto execute:

```bash
streamlit run filas_app.py
```

Como alternativa você pode iniciar o app através do wrapper principal:

```bash
python main.py
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
