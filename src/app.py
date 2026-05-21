import sys
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core import (
    mm1,
    mms,
    mm1k,
    mmsk,
    mm1n,
    mmsn,
    mg1,
    priority_preemptive,
    priority_nonpreemptive,
)
from src.ui import (
    plot_probs,
    metrics_html,
    show_results,
    show_priority_results,
    plot_priority_bars,
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #6a1b6a, #a855c8, #c084dc);
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
section[data-testid="stSidebar"] label {
    color: #b0c4de !important; font-size:0.8rem;
    letter-spacing:0.04em; text-transform:uppercase;
}
.main .block-container { background:#f7f9fc; padding-top:1.5rem; }

.header-card {
    background: linear-gradient(135deg,#e060b0 0%,#b07fd4 60%,#dba8ef 100%);
    border-radius:16px; padding:2rem 2.5rem; color:white;
    margin-bottom:2rem; box-shadow:0 8px 32px rgba(224,96,176,.25);
}
.header-card h1 { font-size:2rem; font-weight:700; margin:0; letter-spacing:-.02em; }
.header-card p  { margin:.3rem 0 0; opacity:.9; font-size:1rem; }

.metric-grid {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr));
    gap:1rem; margin:1.5rem 0;
}
.metric-card {
    background:white; border-radius:12px; padding:1.1rem 1.3rem;
    box-shadow:0 2px 12px rgba(0,0,0,.07); border-left:4px solid #e060b0;
    transition:transform .2s;
}
.metric-card:hover { transform:translateY(-2px); }
.metric-card .label { font-size:.72rem; text-transform:uppercase;
    letter-spacing:.06em; color:#888; font-weight:600; margin-bottom:.3rem; }
.metric-card .value { font-family:'JetBrains Mono',monospace;
    font-size:1.5rem; font-weight:700; color:#1a1a2e; }
.metric-card .unit { font-size:.7rem; color:#aaa; margin-top:.1rem; }

.section-title {
    font-size:.95rem; font-weight:700; letter-spacing:.05em;
    text-transform:uppercase; color:#e060b0;
    border-bottom:2px solid #e060b0; padding-bottom:.4rem; margin:1.5rem 0 1rem;
}
.info-box {
    background:#fdf0f8; border:1px solid #f0b0e0; border-radius:10px;
    padding:1rem 1.2rem; font-size:.86rem; color:#a0307a; margin:1rem 0;
}
.priority-card {
    background:white; border-radius:12px; padding:1.2rem 1.4rem;
    box-shadow:0 2px 12px rgba(0,0,0,.08); margin:0.5rem 0;
}
.priority-card h4 { margin:0 0 .8rem; font-size:.9rem; color:#1a1a2e; }
.prio-badge {
    display:inline-block; background:#e060b0; color:white;
    border-radius:20px; padding:.15rem .7rem; font-size:.75rem;
    font-weight:700; margin-right:.5rem;
}
button[data-baseweb="tab"] {
    font-family:'Space Grotesk',sans-serif !important; font-weight:600 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #e060b0 !important;
}
[data-baseweb="tab-highlight"] {
    background-color: #e060b0 !important;
}
[data-baseweb="tab-border"] {
    background-color: #dba8ef !important;
}

[data-baseweb="slider"] [role="slider"] {
    background-color: #e060b0 !important;
    border-color: #e060b0 !important;
}
[data-baseweb="slider"] div[data-testid="stSlider"] div div div {
    background: #e060b0 !important;
}
div[data-testid="stSlider"] > div > div > div > div {
    background: #e060b0 !important;
}
[data-baseweb="slider"] [data-testid="stSlider"] {
    accent-color: #e060b0 !important;
}
input[type="range"] { accent-color: #e060b0 !important; }
div[class*="StyledThumb"] { background: #e060b0 !important; border-color: #e060b0 !important; }
div[class*="StyledInnerThumb"] { background: white !important; }
div[class*="StyledTrack"][aria-label="track-fill"],
div[class*="StyledTrack-0"] { background: #e060b0 !important; }

.stButton > button { border-color: #e060b0 !important; color: #e060b0 !important; }
.stButton > button:hover { background: #e060b0 !important; color: white !important; }

[data-baseweb="select"] [data-baseweb="select-control"]:focus-within,
[data-baseweb="base-input"]:focus-within {
    border-color: #e060b0 !important;
}

input[type="checkbox"]:checked, input[type="radio"]:checked {
    accent-color: #e060b0 !important;
}
</style>
"""

ALL_MODELS = [
    "M/M/1   — Clássico (1 servidor, fila infinita)",
    "M/M/s   — Clássico (s servidores, fila infinita)",
    "M/M/1/K — Capacidade finita (1 servidor)",
    "M/M/s/K — Capacidade finita (s servidores)",
    "M/M/1/N — População finita (1 servidor)",
    "M/M/s/N — População finita (s servidores)",
    "M/G/1   — Serviço genérico (1 servidor)",
    "Prioridades — Com interrupção",
    "Prioridades — Sem interrupção",
]

MODEL_DESCRIPTIONS = {
    "M/M/1": (
        "Poisson(λ), Exp(μ), 1 servidor, capacidade infinita.",
        ["λ: chegada", "μ: serviço"],
    ),
    "M/M/s": (
        "Poisson(λ), Exp(μ), s servidores, capacidade infinita (Erlang-C).",
        ["λ: chegada", "μ: serviço por servidor", "s: nº servidores"],
    ),
    "M/M/1/K": (
        "Poisson(λ), Exp(μ), 1 servidor, capacidade K.",
        ["λ: chegada", "μ: serviço", "K: capacidade total (fila + em atendimento)"],
    ),
    "M/M/s/K": (
        "Poisson(λ), Exp(μ), s servidores, capacidade K.",
        ["λ: chegada", "μ: serviço por servidor", "s: nº servidores", "K: capacidade total"],
    ),
    "M/M/1/N": (
        "Poisson(λ), Exp(μ), 1 servidor, população finita N.",
        ["λ: chegada individual", "μ: serviço", "N: população total"],
    ),
    "M/M/s/N": (
        "Poisson(λ), Exp(μ), s servidores, população finita N.",
        ["λ: chegada individual", "μ: serviço", "s: nº servidores", "N: população"],
    ),
    "M/G/1": (
        "Poisson(λ), serviço genérico (média 1/μ, variância σ²), 1 servidor.",
        [
            "λ: chegada",
            "μ: taxa serviço",
            "σ²: variância do tempo de serviço",
            "Exp → σ²=1/μ²  |  Determinístico → σ²=0",
        ],
    ),
    "Prioridades": (
        "k classes de prioridade, Poisson(λₖ), Exp(μ), s servidores.",
        [
            "λₖ: chegada por classe",
            "μ: serviço (igual p/ todos)",
            "s: nº servidores",
            "Prioridade 1 = maior prioridade",
        ],
    ),
}

DEFAULT_PRIORITY_LAMBDAS = [0.2, 0.6, 1.2, 0.5, 0.3]


def main():
    st.set_page_config(
        page_title="Teoria das Filas",
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 🎛️ Configuração")
        st.markdown("---")
        model = st.sidebar.selectbox("Modelo de Filas", ALL_MODELS, key="model")
        model_key = model.split("—")[0].strip()

        st.markdown("---")
        st.markdown("### Parâmetros")

        is_priority = model_key.startswith("Prioridades")
        is_mg1 = model_key.startswith("M/G/1")
        is_K = "/K" in model_key
        is_N = "/N" in model_key
        is_mm1_inf = model_key == "M/M/1"
        is_mms_inf = model_key == "M/M/s"
        has_s = ("s/K" in model_key or "s/N" in model_key or is_mms_inf)

        if not is_priority:
            lam = st.number_input("Taxa de chegada (λ)", min_value=0.0001, value=3.0, step=0.0001, format="%.4f")
            mu = st.number_input("Taxa de serviço (μ)", min_value=0.0001, value=4.0, step=0.0001, format="%.4f")
            s = st.number_input("Nº de servidores (s)", min_value=1, value=2, step=1) if has_s else 1
            if is_K:
                K = st.number_input("Capacidade do sistema (K)", min_value=1, value=5, step=1)
                N = None
            elif is_N:
                N = st.number_input("Tamanho da população (N)", min_value=1, value=10, step=1)
                K = None
            else:
                K = None
                N = None
            sigma2 = None
            if is_mg1:
                sigma2 = st.number_input(
                    "Variância do serviço (σ²)",
                    min_value=0.0,
                    value=round(1 / 36, 6),
                    step=0.0001,
                    format="%.6f",
                    help="Para exp: σ²=1/μ². Para determinístico: σ²=0",
                )
        else:
            lam = mu = s = K = N = sigma2 = None
            st.markdown("**Número de classes de prioridade**")
            num_classes = st.number_input("Classes", min_value=2, max_value=6, value=3, step=1)
            mu_p = st.number_input("Taxa de serviço (μ) — igual p/ todas", min_value=0.0001, value=3.0, step=0.0001, format="%.4f")
            s_p = st.number_input("Nº de servidores (s)", min_value=1, value=1, step=1)
            lambdas = []
            for k in range(int(num_classes)):
                lk = st.number_input(
                    f"λ{k+1} (prioridade {k+1})",
                    min_value=0.0001,
                    value=DEFAULT_PRIORITY_LAMBDAS[k],
                    step=0.0001,
                    format="%.4f",
                    key=f"lk{k}",
                )
                lambdas.append(lk)

        time_unit = st.sidebar.selectbox("Unidade de tempo", ["min", "h", "s", "dias"], key="time_unit")

    st.markdown(
        """
<div class="header-card">
  <h1>📐 Teoria das Filas</h1>
  <p>Modelos M/M/1, M/M/s, M/G/1 e Prioridades — P108 Otimização II · Inatel</p>
</div>""",
        unsafe_allow_html=True,
    )

    tab_calc, tab_theory, tab_sens = st.tabs(["🧮 Calculadora", "📖 Teoria & Fórmulas", "📊 Análise de Sensibilidade"])

    with tab_calc:
        d = MODEL_DESCRIPTIONS.get("Prioridades" if is_priority else model_key, ("", []))
        st.markdown(
            f"""
    <div class="info-box">
        <strong>{model}</strong><br><br>{d[0]}<br><br>
        <b>Parâmetros:</b> {'  ·  '.join(f'<code>{p}</code>' for p in d[1])}
    </div>""",
            unsafe_allow_html=True,
        )

        try:
            if model_key == "M/M/1":
                res = mm1(lam, mu)
                if res is None:
                    st.error(f"ρ = {lam/mu:.3f} ≥ 1. Sistema instável.")
                else:
                    show_results(res, f"M/M/1  λ={lam} μ={mu}", time_unit)

            elif model_key == "M/M/s":
                s_v = int(s)
                res = mms(lam, mu, s_v)
                if res is None:
                    st.error(f"ρ = {lam/(s_v*mu):.3f} ≥ 1. Sistema instável.")
                else:
                    show_results(res, f"M/M/{s_v}  λ={lam} μ={mu} s={s_v}", time_unit)
                    st.markdown(
                        f"""<div class="info-box">
                    <b>Erlang-C (C)</b> = probabilidade de espera = {res['C']:.5f}
                </div>""",
                        unsafe_allow_html=True,
                    )

            elif model_key == "M/M/1/K":
                res = mm1k(lam, mu, int(K))
                show_results(res, f"M/M/1/K  λ={lam} μ={mu} K={int(K)}", time_unit)

            elif model_key == "M/M/s/K":
                s_v, K_v = int(s), int(K)
                if K_v < s_v:
                    st.error("K deve ser ≥ s.")
                else:
                    res = mmsk(lam, mu, s_v, K_v)
                    show_results(res, f"M/M/{s_v}/K  λ={lam} μ={mu} s={s_v} K={K_v}", time_unit)

            elif model_key == "M/M/1/N":
                res = mm1n(lam, mu, int(N))
                show_results(res, f"M/M/1/N  λ={lam} μ={mu} N={int(N)}", time_unit)

            elif model_key == "M/M/s/N":
                s_v, N_v = int(s), int(N)
                res = mmsn(lam, mu, s_v, N_v)
                show_results(res, f"M/M/{s_v}/N  λ={lam} μ={mu} s={s_v} N={N_v}", time_unit)

            elif is_mg1:
                res = mg1(lam, mu, sigma2)
                if res is None:
                    st.error(f"ρ = {lam/mu:.3f} ≥ 1. Sistema instável.")
                else:
                    st.markdown(
                        f'<div class="section-title">📊 M/G/1 — λ={lam} μ={mu} σ²={sigma2:.5f}</div>',
                        unsafe_allow_html=True,
                    )
                    items = [
                        ("P₀", f"{res['P0']:.5f}", "prob. sistema vazio"),
                        ("ρ", f"{res['rho']:.4f}", "utilização"),
                        ("L", f"{res['L']:.4f}", "clientes no sistema"),
                        ("Lq", f"{res['Lq']:.4f}", "clientes na fila"),
                        ("W", f"{res['W']:.4f}", f"tempo no sistema ({time_unit})"),
                        ("Wq", f"{res['Wq']:.4f}", f"espera na fila ({time_unit})"),
                    ]
                    st.markdown(metrics_html(items), unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("#### Comparação com M/M/1 (σ²=1/μ²)")
                    sig2_mm1 = 1 / mu**2
                    res_mm1 = mg1(lam, mu, sig2_mm1)
                    if res_mm1:
                        ratio = res['Lq'] / res_mm1['Lq'] if res_mm1['Lq'] > 0 else float('nan')
                        cols = st.columns(3)
                        cols[0].metric("Lq (M/G/1)", f"{res['Lq']:.4f}")
                        cols[1].metric("Lq (M/M/1)", f"{res_mm1['Lq']:.4f}")
                        cols[2].metric("Razão Lq", f"{ratio:.4f}", help="Lq(M/G/1) / Lq(M/M/1)")

                        sig_vals = np.linspace(0, 2 / mu**2, 60)
                        lqs = [(lam**2 * sv + res['rho']**2) / (2 * (1 - res['rho'])) for sv in sig_vals]
                        fig_g, ax = plt.subplots(figsize=(8, 3))
                        ax.plot(sig_vals, lqs, color='#e060b0', lw=2.5)
                        ax.axvline(sigma2, color='#a050c8', ls='--', lw=1.5, label=f'σ²={sigma2:.4f}')
                        ax.axvline(1 / mu**2, color='#dba8ef', ls='--', lw=1.5, label='σ²=1/μ² (Exp)')
                        ax.set_xlabel("σ²")
                        ax.set_ylabel("Lq")
                        ax.set_title("Sensibilidade de Lq à variância do serviço")
                        ax.legend()
                        ax.grid(alpha=.3)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        fig_g.patch.set_facecolor('#f7f9fc')
                        ax.set_facecolor('#f7f9fc')
                        plt.tight_layout()
                        st.pyplot(fig_g, use_container_width=True)

            elif is_priority:
                preemptive = "Com interrupção" in model
                lam_tot = sum(lambdas)
                rho_tot = lam_tot / (s_p * mu_p)
                if rho_tot >= 1:
                    st.error(f"ρ total = {rho_tot:.3f} ≥ 1. Sistema instável.")
                else:
                    results = (
                        priority_preemptive(lambdas, mu_p, int(s_p))
                        if preemptive
                        else priority_nonpreemptive(lambdas, mu_p, int(s_p))
                    )
                    tipo = "Com interrupção (Preemptivo)" if preemptive else "Sem interrupção (Não-preemptivo)"
                    st.markdown(
                        f'<div class="section-title">📊 Prioridades — {tipo} | s={int(s_p)} μ={mu_p}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                <div class="info-box">
                  λ total = {lam_tot:.3f} · μ = {mu_p} · s = {int(s_p)} · ρ total = {rho_tot:.4f}
                </div>""",
                        unsafe_allow_html=True,
                    )
                    show_priority_results(results, time_unit)
                    st.pyplot(plot_priority_bars(results, time_unit), use_container_width=True)

                    st.markdown("---")
                    df_prio = pd.DataFrame([
                        {
                            "Prioridade": f"P{r['k']}",
                            "λₖ": r['lam'],
                            "ρₖ": f"{r['rho']:.4f}",
                            "W": f"{r['W']:.5f}",
                            "Wq": f"{r['Wq']:.5f}",
                            "L": f"{r['L']:.5f}",
                            "Lq": f"{r['Lq']:.5f}",
                        }
                        for r in results
                    ])
                    st.dataframe(df_prio, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Erro: {e}")
            import traceback
            st.code(traceback.format_exc())

    with tab_theory:
        st.markdown("### Resumo dos Modelos")

        with st.expander("📌 M/M/1 — Clássico (1 servidor, fila infinita)"):
            st.markdown(
                """
- **ρ = λ/μ** (deve ser < 1 para estabilidade)
- **P₀ = 1 − ρ** · **Pₙ = (1−ρ)·ρⁿ** para n = 0,1,2,…
- **L = ρ/(1−ρ)** · **Lq = ρ²/(1−ρ)**
- **W = 1/(μ−λ)** · **Wq = ρ/(μ−λ)**
                """
            )
        with st.expander("📌 M/M/s — Clássico (s servidores, fila infinita)"):
            st.markdown(
                """
- **ρ = λ/(sμ)** (deve ser < 1) · **r = λ/μ**
- **P₀ = [Σₙ₌₀ˢ⁻¹ rⁿ/n! + rˢ/(s!·(1−ρ))]⁻¹**
- **C (Erlang-C)** = prob. de espera = **[rˢ/(s!·(1−ρ))]·P₀**
- **Lq = C·ρ/(1−ρ)** · **L = Lq + r**
- **Wq = Lq/λ** · **W = Wq + 1/μ**
                """
            )
        with st.expander("📌 M/M/1/K — Capacidade finita, 1 servidor"):
            st.markdown(
                """
- **ρ = λ/μ** · **P₀ = (1−ρ)/(1−ρᴷ⁺¹)**
- **Pₙ = P₀·ρⁿ** para n=1,…,K
- **L = ρ/(1−ρ) − (K+1)ρᴷ⁺¹/(1−ρᴷ⁺¹)**  ·  **Lq = L − (1−P₀)**
- **λ̄ = λ(1−Pₖ)**  ·  **W = L/λ̄**  ·  **Wq = Lq/λ̄**
                """
            )
        with st.expander("📌 M/M/s/K — Capacidade finita, s servidores"):
            st.markdown(
                """
- **ρ = λ/(sμ)** · P₀ calculada via dupla somatória
- Pₙ = (λ/μ)ⁿ/n!·P₀ para n≤s  |  Pₙ = (λ/μ)ⁿ/(s!·sⁿ⁻ˢ)·P₀ para s<n≤K
- **Lq** via fórmula fechada com ρᴷ⁻ˢ
- L = Σ nPₙ + Lq + s(1−Σ Pₙ), n<s
                """
            )
        with st.expander("📌 M/M/1/N — População finita, 1 servidor"):
            st.markdown(
                """
- Taxa efetiva = (N−n)λ quando há n no sistema
- **P₀ = 1/Σ[N!/(N−n)!·(λ/μ)ⁿ]**
- **L = N − (μ/λ)(1−P₀)**  ·  **Lq = N − ((λ+μ)/λ)(1−P₀)**
- **λ̄ = λ(N−L)**
                """
            )
        with st.expander("📌 M/M/s/N — População finita, s servidores"):
            st.markdown(
                """
- Pₙ análogo ao M/M/1/N mas com fator s! e sⁿ⁻ˢ para n≥s
- **L = Σ n·Pₙ**  ·  **Lq = L − (λ/μ)(N−L)**  ·  **λ̄ = λ(N−L)**
                """
            )
        with st.expander("📌 M/G/1 — Fórmula de Pollaczek-Khintchine"):
            st.markdown(
                """
- Chegadas Poisson(λ), serviço qualquer distribuição com média **1/μ** e variância **σ²**
- **ρ = λ/μ**  ·  **P₀ = 1 − ρ**
- **Lq = (λ²σ² + ρ²) / (2(1−ρ))**  ← fórmula P-K
- **L = ρ + Lq**  ·  **Wq = Lq/λ**  ·  **W = Wq + 1/μ**

Casos especiais:
| Distribuição | σ² | Lq |
|---|---|---|
| Exponencial (M/M/1) | 1/μ² | ρ²/(1−ρ) |
| Determinístico (M/D/1) | 0 | ρ²/(2(1−ρ)) |

> **Lq(M/D/1) = ½ · Lq(M/M/1)** sempre!
                """
            )
        with st.expander("📌 Prioridades — Com interrupção (Preemptivo)"):
            st.markdown(
                """
O cliente em atendimento é **interrompido** quando chega um de prioridade maior.

**W_k = (1/μ) / [(1 − A_{k−1})(1 − A_k)]**

onde **A_k = Σᵢ₌₁ᵏ ρᵢ** e **ρᵢ = λᵢ/(sμ)**

- Lq_k = L_k − λ_k/μ  ·  Wq_k = W_k − 1/μ  ·  L_k = λ_k · W_k
                """
            )
        with st.expander("📌 Prioridades — Sem interrupção (Não-preemptivo)"):
            st.markdown(
                """
O cliente em atendimento **conclui** antes de ceder lugar a um de maior prioridade.

**W_k = 1 / [(s!·(sμ−λ)/rˢ·Σrʲ/j! + sμ)·(1−A_{k−1})·(1−A_k)] + 1/μ**

onde **r = λ/μ** e **A_k = Σᵢ₌₁ᵏ ρᵢ**

Para s=1: o termo base se simplifica diretamente.
                """
            )
        with st.expander("📖 Referências"):
            st.markdown(
                """
- Hillier & Lieberman, *Introdução à Pesquisa Operacional*, 9ª ed., McGraw-Hill, 2013.
- Arenales et al., *Pesquisa Operacional*, 6ª ed., Campus, 2007.
- Andrade, E.L., *Introdução a Pesquisa Operacional*, 4ª ed., LTC, 2009.
- Taha, H.A., *Pesquisa Operacional*, 8ª ed., Pearson, 2008.
                """
            )

    with tab_sens:
        st.markdown("### Análise de Sensibilidade")
        c1, c2, c3 = st.columns(3)
        with c1:
            sm = st.selectbox("Modelo", ["M/M/1/K", "M/M/s/K", "M/M/1/N", "M/M/s/N"], key="sm")
        with c2:
            sl = st.number_input("λ", value=3.0, min_value=0.0001, step=0.0001, format="%.4f", key="sl2")
            smu = st.number_input("μ", value=4.0, min_value=0.0001, step=0.0001, format="%.4f", key="smu2")
        with c3:
            ss = st.number_input("s", value=2, min_value=1, step=1, key="ss2") if "s/" in sm else 1
            prng = st.slider("Faixa K/N", 2, 40, (2, 20))

        rows = []
        for kn in range(prng[0], prng[1] + 1):
            try:
                if sm == "M/M/1/K":
                    r = mm1k(sl, smu, kn)
                elif sm == "M/M/s/K":
                    r = mmsk(sl, smu, int(ss), kn) if kn >= int(ss) else None
                elif sm == "M/M/1/N":
                    r = mm1n(sl, smu, kn)
                else:
                    r = mmsn(sl, smu, int(ss), kn) if kn >= int(ss) else None
                if r:
                    rows.append({"K/N": kn, "L": r['L'], "Lq": r['Lq'], "W": r['W'], "Wq": r['Wq'], "P0": r['P0']})
            except Exception:
                pass

        if rows:
            df_s = pd.DataFrame(rows)
            fig3, axs = plt.subplots(1, 3, figsize=(14, 3.5))
            for ax, (y1, y2, lbl) in zip(
                axs,
                [
                    ("L", "Lq", "Clientes"),
                    ("W", "Wq", f"Tempo ({time_unit})"),
                    ("P0", None, "P₀"),
                ],
            ):
                ax.plot(df_s["K/N"], df_s[y1], 'o-', color='#e060b0', label=y1, lw=2)
                if y2:
                    ax.plot(df_s["K/N"], df_s[y2], 's--', color='#b07fd4', label=y2, lw=1.5)
                ax.set_xlabel("K/N")
                ax.set_ylabel(lbl)
                ax.legend(fontsize=8)
                ax.grid(alpha=.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_facecolor('#f7f9fc')
            fig3.patch.set_facecolor('#f7f9fc')
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            st.dataframe(df_s.style.format({c: "{:.4f}" for c in df_s.columns if c != "K/N"}), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
