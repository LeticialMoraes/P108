import math
import sys
import unicodedata
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
from src.conversion import (
    DURATION_UNIT_LABEL,
    RATE_UNIT_LABEL,
    SECONDS_PER,
    compound_duration_to_seconds,
    duration_to_seconds,
    mu_per_second_from_mean_service,
    rate_in_period,
    rate_per_second,
    scale_result_times_for_display,
    variance_to_seconds_squared,
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
.priority-card h4 { margin:0 .8rem; font-size:.9rem; color:#1a1a2e; }
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

# Rótulos exatos do select (usados na comparação de preemptivo)
PRIORITY_PREEMPT_LABEL = "Prioridades — Com interrupção"
PRIORITY_NONPREEMPT_LABEL = "Prioridades — Sem interrupção"

ALL_MODELS = [
    "M/M/1   — Clássico (1 servidor, fila infinita)",
    "M/M/s   — Clássico (s servidores, fila infinita)",
    "M/M/1/K — Capacidade finita (1 servidor)",
    "M/M/s/K — Capacidade finita (s servidores)",
    "M/M/1/N — População finita (1 servidor)",
    "M/M/s/N — População finita (s servidores)",
    "M/G/1   — Serviço genérico (1 servidor)",
    PRIORITY_PREEMPT_LABEL,
    PRIORITY_NONPREEMPT_LABEL,
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

# Rótulos da barra lateral (chegadas / serviço) — só número + unidade
_SID_LAM_TAXA = "Informar λ"
_SID_LAM_TEMPO = "Informar tempo médio entre chegadas"
_SID_MU_TAXA = "Informar μ"
_SID_MU_TEMPO = "Informar tempo médio de serviço"

_SID_TEMPO_SIMPLES = "Tempo simples (valor + unidade)"
_SID_TEMPO_COMPOSTO = "Tempo composto (dias / horas / minutos / segundos)"

_COMPOUND_LABEL_BOX = (
    "min-height:3.1rem;display:flex;align-items:flex-end;"
    "font-size:0.76rem;line-height:1.12;margin:0 0 0.35rem 0;"
)


def _compound_time_four_columns(key_prefix: str):
    """
    Quatro number_inputs em colunas com rótulos visuais alinhados (sidebar estreita).
    """
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div style="{_COMPOUND_LABEL_BOX}">Dias</div>',
            unsafe_allow_html=True,
        )
        d = st.number_input(
            f"Dias ({key_prefix})",
            min_value=0.0,
            value=0.0,
            step=0.25,
            format="%.4f",
            key=f"{key_prefix}_d",
            label_visibility="collapsed",
        )
    with c2:
        st.markdown(
            f'<div style="{_COMPOUND_LABEL_BOX}">Horas</div>',
            unsafe_allow_html=True,
        )
        h = st.number_input(
            f"Horas ({key_prefix})",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.4f",
            key=f"{key_prefix}_h",
            label_visibility="collapsed",
        )
    with c3:
        st.markdown(
            f'<div style="{_COMPOUND_LABEL_BOX}">Minutos</div>',
            unsafe_allow_html=True,
        )
        m = st.number_input(
            f"Minutos ({key_prefix})",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.4f",
            key=f"{key_prefix}_m",
            label_visibility="collapsed",
        )
    with c4:
        st.markdown(
            f'<div style="{_COMPOUND_LABEL_BOX}">Segundos</div>',
            unsafe_allow_html=True,
        )
        s = st.number_input(
            f"Segundos ({key_prefix})",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.4f",
            key=f"{key_prefix}_s",
            label_visibility="collapsed",
        )
    return d, h, m, s


def _fmt_pt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def _html_duration_interpreted(sec: float) -> str:
    if sec is None or sec <= 0:
        return ""
    m = sec / 60.0
    h = sec / 3600.0
    return (
        "<b>Interpretado como:</b><br>"
        f"· <b>{_fmt_pt(sec, 2)}</b> segundos<br>"
        f"· <b>{_fmt_pt(m, 6)}</b> minutos<br>"
        f"· <b>{_fmt_pt(h, 6)}</b> horas"
    )


def _html_rho_review(model_key: str, lam_sec: float, mu_sec: float, s: int) -> str:
    if model_key == "M/M/1/N":
        rho = lam_sec / mu_sec if mu_sec else float("nan")
        expr = "ρ = λ/μ"
    else:
        rho = lam_sec / (s * mu_sec) if mu_sec else float("nan")
        expr = "ρ = λ/(sμ)"
    return (
        f'<div class="info-box" style="margin-top:0.75rem"><b>Utilização ({expr})</b><br>'
        f"ρ = <b>{_fmt_pt(rho, 6)}</b> "
        "<span style='opacity:0.85'>(λ e μ na base interna 1/s; mesmo valor adimensional)</span>"
        "</div>"
    )


def _render_diagnosis_expander(res_raw: dict, res: dict, time_unit: str) -> None:
    """Expander: ΣPn, λ̄, tabela Pn, estados; Lei de Little."""
    pn = res_raw.get("Pn")
    if not pn:
        return
    sum_pn = sum(pn)
    lam_eff_s = res_raw.get("lam_eff")
    n_states = len(pn)

    with st.expander("Diagnóstico e Validação", expanded=False):
        st.markdown(f"**ΣPₙ** = {_fmt_pt(sum_pn, 8)}  (esperado ≈ 1)")
        if lam_eff_s is not None:
            le_m = rate_in_period(lam_eff_s, "min")
            le_h = rate_in_period(lam_eff_s, "h")
            st.markdown(
                f"**λ̄ (taxa efetiva de entrada no sistema)** = {_fmt_pt(le_m)}/min · {_fmt_pt(le_h)}/h "
                f"· {_fmt_pt(rate_in_period(lam_eff_s, 's'), 8)}/s"
            )
        st.markdown(f"**Quantidade de estados** (n = 0 … {n_states - 1}): **{n_states}**")

        df = pd.DataFrame(
            {
                "n": list(range(n_states)),
                "P(n)": [f"{p:.8f}" for p in pn],
                "%": [f"{p * 100:.5f}%" for p in pn],
            }
        )
        st.dataframe(df, use_container_width=True, hide_index=True, height=min(400, 60 + 28 * n_states))

        st.markdown("---")
        st.markdown("**Lei de Little** (consistência; mesma unidade de tempos dos resultados)")
        lam_bar = res.get("lam_eff")
        L = res.get("L")
        W = res.get("W")
        Lq = res.get("Lq")
        Wq = res.get("Wq")

        if lam_bar is not None and W is not None and L is not None:
            try:
                lb, w, l = float(lam_bar), float(W), float(L)
                if math.isfinite(lb) and math.isfinite(w) and math.isfinite(l):
                    prod = lb * w
                    err = abs(l - prod)
                    st.markdown(
                        f"- **L** = {_fmt_pt(l, 6)} · **λ̄×W** = {_fmt_pt(prod, 6)} · "
                        f"**erro absoluto** = {_fmt_pt(err, 8)}"
                    )
            except (TypeError, ValueError):
                st.caption("L / Lei de Little: valores não numéricos.")
        if lam_bar is not None and Wq is not None and Lq is not None:
            try:
                lb, wq, lq = float(lam_bar), float(Wq), float(Lq)
                if math.isfinite(lb) and math.isfinite(wq) and math.isfinite(lq):
                    prod = lb * wq
                    err = abs(lq - prod)
                    st.markdown(
                        f"- **Lq** = {_fmt_pt(lq, 6)} · **λ̄×Wq** = {_fmt_pt(prod, 6)} · "
                        f"**erro absoluto** = {_fmt_pt(err, 8)}"
                    )
            except (TypeError, ValueError):
                st.caption("Lq / Lei de Little: valores não numéricos.")


def _html_detailed_input_conference(audit: dict, lam_sec: float, mu_sec: float) -> str:
    parts = ['<div class="info-box"><b>Conferência das entradas</b><br><br>']

    parts.append("<b>Chegadas</b><br>")
    if audit.get("entrada_chegadas") in (_SID_LAM_TAXA, "Taxa λ"):
        lam, ul = audit["lam"], audit["unidade_lam"]
        parts.append(f"<b>Valor informado:</b> λ = {_fmt_pt(lam)} ({RATE_UNIT_LABEL[ul]})<br>")
        parts.append("<i>Equivale a tempo médio entre chegadas</i> = 1/λ:<br>")
    else:
        et = audit.get("tbc_entrada", _SID_TEMPO_SIMPLES)
        if et == _SID_TEMPO_COMPOSTO:
            d, h, m, s = audit["tbc_d"], audit["tbc_h"], audit["tbc_m"], audit["tbc_s"]
            parts.append(
                f"<b>Tempo composto (entre chegadas):</b> {_fmt_pt(d)} d + {_fmt_pt(h)} h + "
                f"{_fmt_pt(m)} min + {_fmt_pt(s)} s<br>"
            )
            parts.append("<b>Tempo médio entre chegadas</b> (soma, interpretado):<br>")
        else:
            te, ut = audit["tempo_entre_chegadas"], audit["unidade_tbc"]
            parts.append(
                f"<b>Valor informado:</b> {_fmt_pt(te)} {DURATION_UNIT_LABEL[ut]} (tempo médio entre chegadas)<br>"
            )
            parts.append("<b>Tempo médio entre chegadas</b> (interpretado):<br>")
    if lam_sec and lam_sec > 0:
        parts.append(_html_duration_interpreted(1.0 / lam_sec) + "<br><br>")
    parts.append("<b>λ calculado</b> (usado nos cálculos):<br>")
    parts.append(
        f"· {_fmt_pt(rate_in_period(lam_sec, 'min'))}/min · {_fmt_pt(rate_in_period(lam_sec, 'h'))}/h<br><br>"
    )

    parts.append("<b>Serviço</b><br>")
    if audit.get("mu_path") == "taxa":
        mu, um = audit["mu"], audit["unidade_mu"]
        parts.append(f"<b>Valor informado:</b> μ = {_fmt_pt(mu)} ({RATE_UNIT_LABEL[um]})<br>")
        parts.append("<i>Equivale a tempo médio de serviço</i> = 1/μ:<br>")
    else:
        es_ent = audit.get("es_entrada", _SID_TEMPO_SIMPLES)
        if es_ent == _SID_TEMPO_COMPOSTO:
            d, h, m, s = audit["es_d"], audit["es_h"], audit["es_m"], audit["es_s"]
            parts.append(
                f"<b>Tempo composto (serviço):</b> {_fmt_pt(d)} d + {_fmt_pt(h)} h + "
                f"{_fmt_pt(m)} min + {_fmt_pt(s)} s<br>"
            )
            parts.append("<b>Tempo médio de serviço</b> (soma, interpretado):<br>")
        else:
            es, ue = audit["es"], audit["unidade_es"]
            parts.append(
                f"<b>Valor informado:</b> {_fmt_pt(es)} {DURATION_UNIT_LABEL.get(ue, ue)} (tempo médio de serviço)<br>"
            )
            parts.append("<b>Tempo médio de serviço</b> (interpretado):<br>")
    if mu_sec and mu_sec > 0:
        parts.append(_html_duration_interpreted(1.0 / mu_sec) + "<br><br>")
    parts.append("<b>μ calculado</b> (usado nos cálculos):<br>")
    parts.append(f"· {_fmt_pt(rate_in_period(mu_sec, 'min'))}/min · {_fmt_pt(rate_in_period(mu_sec, 'h'))}/h")
    parts.append("</div>")
    return "".join(parts)


def main():
    st.set_page_config(
        page_title="Teoria das Filas",
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CSS, unsafe_allow_html=True)

    # ── session_state defaults ───────────────────────────────────────────────
    if "model_idx" not in st.session_state:
        st.session_state.model_idx = 0

    input_audit = None

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎛️ Configuração")
        st.markdown("---")

        # on_change atualiza model_idx apenas quando o dropdown fecha
        def _on_model_change():
            st.session_state.model_idx = ALL_MODELS.index(st.session_state._model_select)

        st.selectbox(
            "Modelo de Filas",
            ALL_MODELS,
            index=st.session_state.model_idx,
            key="_model_select",
            on_change=_on_model_change,
        )

        # Lê o modelo confirmado (não muda durante a abertura do dropdown)
        model = ALL_MODELS[st.session_state.model_idx]
        model_key = model.split("—")[0].strip()

        st.markdown("---")
        st.markdown("### Parâmetros")

        is_priority = model_key.startswith("Prioridades")
        is_mg1 = model_key.startswith("M/G/1")
        is_K = "/K" in model_key
        is_N = "/N" in model_key
        is_mms_inf = model_key == "M/M/s"
        has_s = ("s/K" in model_key or "s/N" in model_key or is_mms_inf)

        time_unit = st.selectbox(
            "Unidade para tempos (W, Wq) nos resultados",
            ["min", "h", "s", "dias"],
            key="time_unit",
            help="Tempos médios exibidos nesta unidade. λ e μ podem ser informados em outras unidades abaixo.",
        )

        if not is_priority:
            st.markdown("**Chegadas**")
            entrada_chegadas = st.radio(
                "Modo",
                [_SID_LAM_TAXA, _SID_LAM_TEMPO],
                horizontal=True,
                key="entrada_chegadas",
                help="λ com unidade **ou** tempo médio entre chegadas (simples: valor+unidade; composto: d/h/min/s).",
            )

            lam = None
            unidade_lam = None
            tempo_entre_chegadas = None
            unidade_tbc = None

            if entrada_chegadas == _SID_LAM_TAXA:
                unidade_lam = st.selectbox(
                    "Unidade da taxa λ",
                    ["s", "min", "h", "dias"],
                    format_func=lambda u: RATE_UNIT_LABEL[u],
                    index=1,
                    key="unidade_lam",
                    help="Ex.: 1 **por hora** → escolha «por hora» e digite 1.",
                )
                lam = st.number_input(
                    f"Taxa de chegada λ ({RATE_UNIT_LABEL[unidade_lam]})",
                    min_value=0.0001,
                    value=3.0,
                    step=0.0001,
                    format="%.4f",
                )
                _lam_sec = rate_per_second(lam, unidade_lam)
                st.caption(
                    f"Equiv.: **{rate_in_period(_lam_sec, 'h'):.4f}/h** · "
                    f"**{rate_in_period(_lam_sec, 'min'):.4f}/min** · "
                    f"{rate_in_period(_lam_sec, 's'):.6f}/s"
                )
            else:
                entrada_tbc = st.radio(
                    "Formato do tempo",
                    [_SID_TEMPO_SIMPLES, _SID_TEMPO_COMPOSTO],
                    horizontal=True,
                    key="entrada_tbc",
                    help="Simples: um valor e uma unidade. Composto: some dias, horas, minutos e segundos.",
                )
                tempo_entre_chegadas = None
                unidade_tbc = None
                tbc_d = tbc_h = tbc_m = tbc_s = 0.0
                if entrada_tbc == _SID_TEMPO_SIMPLES:
                    unidade_tbc = st.selectbox(
                        "Unidade do tempo médio entre chegadas",
                        ["s", "min", "h", "dias"],
                        format_func=lambda u: DURATION_UNIT_LABEL[u],
                        index=1,
                        key="unidade_tbc",
                    )
                    tempo_entre_chegadas = st.number_input(
                        f"Tempo médio entre chegadas ({DURATION_UNIT_LABEL[unidade_tbc]})",
                        min_value=0.0001,
                        value=0.3333,
                        step=0.0001,
                        format="%.4f",
                        help="Internamente: λ = 1 / (este tempo), com tempo convertido para segundos.",
                    )
                else:
                    st.caption("Some os campos; a soma deve ser **> 0** (decimais permitidos, ex.: 0,5 dias).")
                    tbc_d, tbc_h, tbc_m, tbc_s = _compound_time_four_columns("tbc")

            lam_sec = None
            try:
                if entrada_chegadas == _SID_LAM_TAXA:
                    lam_sec = rate_per_second(lam, unidade_lam)
                else:
                    if entrada_tbc == _SID_TEMPO_SIMPLES:
                        lam_sec = 1.0 / duration_to_seconds(tempo_entre_chegadas, unidade_tbc)
                    else:
                        lam_sec = 1.0 / compound_duration_to_seconds(tbc_d, tbc_h, tbc_m, tbc_s)
            except (TypeError, ValueError, ZeroDivisionError) as exc_lam:
                st.warning(f"Chegadas: {exc_lam}")

            st.markdown("**Serviço**")
            modo_servico = st.radio(
                "Modo",
                [_SID_MU_TAXA, _SID_MU_TEMPO],
                horizontal=True,
                key="modo_servico",
                help="μ com unidade **ou** tempo médio de serviço (simples: valor+unidade; composto: d/h/min/s).",
            )
            mu_sec = None
            mu = None
            unidade_mu = None
            es = None
            unidade_es = None

            if modo_servico == _SID_MU_TAXA:
                _idx_mu = (
                    ["s", "min", "h", "dias"].index(unidade_lam)
                    if unidade_lam is not None
                    else 1
                )
                unidade_mu = st.selectbox(
                    "Unidade da taxa μ",
                    ["s", "min", "h", "dias"],
                    format_func=lambda u: RATE_UNIT_LABEL[u],
                    index=_idx_mu,
                    key="unidade_mu",
                )
                mu = st.number_input(
                    f"Taxa de serviço μ ({RATE_UNIT_LABEL[unidade_mu]})",
                    min_value=0.0001,
                    value=4.0,
                    step=0.0001,
                    format="%.4f",
                    key="mu_taxa",
                )
                mu_sec = rate_per_second(mu, unidade_mu)
            else:
                entrada_es = st.radio(
                    "Formato do tempo",
                    [_SID_TEMPO_SIMPLES, _SID_TEMPO_COMPOSTO],
                    horizontal=True,
                    key="entrada_es",
                    help="Simples: um valor e uma unidade. Composto: some dias, horas, minutos e segundos.",
                )
                es = None
                unidade_es = None
                es_d = es_h = es_m = es_s = 0.0
                if entrada_es == _SID_TEMPO_SIMPLES:
                    unidade_es = st.selectbox(
                        "Unidade do tempo médio de serviço",
                        ["s", "min", "h", "dias"],
                        format_func=lambda u: DURATION_UNIT_LABEL[u],
                        index=1,
                        key="unidade_es",
                    )
                    es = st.number_input(
                        f"Tempo médio de serviço ({DURATION_UNIT_LABEL[unidade_es]})",
                        min_value=0.0001,
                        value=0.25,
                        step=0.0001,
                        format="%.4f",
                        key="mu_tempo",
                        help="Internamente: μ = 1 / (este tempo), com tempo convertido para segundos.",
                    )
                else:
                    st.caption("Some os campos; a soma deve ser **> 0** (decimais permitidos, ex.: 0,5 dias).")
                    es_d, es_h, es_m, es_s = _compound_time_four_columns("es")
                try:
                    if entrada_es == _SID_TEMPO_SIMPLES:
                        mu_sec = mu_per_second_from_mean_service(es, unidade_es)
                    else:
                        mu_sec = 1.0 / compound_duration_to_seconds(es_d, es_h, es_m, es_s)
                except ValueError as exc_mu:
                    st.warning(f"Serviço: {exc_mu}")
                if mu_sec is not None:
                    if unidade_lam is not None:
                        mu_equiv_lam = rate_in_period(mu_sec, unidade_lam)
                        st.caption(
                            f"μ equivalente ≈ **{mu_equiv_lam:.6f}** ({RATE_UNIT_LABEL[unidade_lam]}) "
                            f"· **{rate_in_period(mu_sec, time_unit):.6f}** ({RATE_UNIT_LABEL[time_unit]})"
                        )
                    else:
                        st.caption(
                            f"μ na unidade dos resultados: **{rate_in_period(mu_sec, time_unit):.6f}** "
                            f"({RATE_UNIT_LABEL[time_unit]})"
                        )
                    mu = rate_in_period(mu_sec, time_unit)

            lam_title = rate_in_period(lam_sec, time_unit) if lam_sec is not None else None
            mu_title = rate_in_period(mu_sec, time_unit) if mu_sec is not None else None

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
            unidade_sigma2 = "min"
            if is_mg1:
                unidade_sigma2 = st.selectbox(
                    "Unidade de σ² (variância do tempo de serviço)",
                    ["s", "min", "h", "dias"],
                    format_func=lambda u: f"({DURATION_UNIT_LABEL[u]})²",
                    index=1,
                    key="unidade_sigma2",
                )
                sigma2 = st.number_input(
                    f"Variância σ² em ({DURATION_UNIT_LABEL[unidade_sigma2]})²",
                    min_value=0.0,
                    value=round(1 / 36, 6),
                    step=0.0001,
                    format="%.6f",
                    help="Exponencial: σ² = (1/μ)² na mesma base que E[S]. Ex.: se μ em 1/min, σ² em min².",
                )

            input_audit = {
                "entrada_chegadas": entrada_chegadas,
                "modo_servico": modo_servico,
            }
            if entrada_chegadas == _SID_LAM_TAXA:
                input_audit["lam"] = lam
                input_audit["unidade_lam"] = unidade_lam
            else:
                input_audit["tbc_entrada"] = entrada_tbc
                if entrada_tbc == _SID_TEMPO_SIMPLES:
                    input_audit["tempo_entre_chegadas"] = tempo_entre_chegadas
                    input_audit["unidade_tbc"] = unidade_tbc
                else:
                    input_audit["tbc_d"] = tbc_d
                    input_audit["tbc_h"] = tbc_h
                    input_audit["tbc_m"] = tbc_m
                    input_audit["tbc_s"] = tbc_s
            if modo_servico == _SID_MU_TAXA:
                input_audit["mu_path"] = "taxa"
                input_audit["mu"] = mu
                input_audit["unidade_mu"] = unidade_mu
            else:
                input_audit["mu_path"] = "tempo"
                input_audit["es_entrada"] = entrada_es
                if entrada_es == _SID_TEMPO_SIMPLES:
                    input_audit["es"] = es
                    input_audit["unidade_es"] = unidade_es
                else:
                    input_audit["es_d"] = es_d
                    input_audit["es_h"] = es_h
                    input_audit["es_m"] = es_m
                    input_audit["es_s"] = es_s
        else:
            lam = mu = s = K = N = sigma2 = None
            lam_sec = mu_sec = None
            lam_title = mu_title = None
            unidade_sigma2 = "min"

            st.markdown("**Número de classes de prioridade**")
            num_classes = st.number_input("Classes", min_value=2, max_value=6, value=3, step=1)

            unidade_lam_p = st.selectbox(
                "Unidade das taxas λₖ",
                ["s", "min", "h", "dias"],
                format_func=lambda u: RATE_UNIT_LABEL[u],
                index=1,
                key="unidade_lam_p",
                help="Mesma regra: «por minuto» com 1 significa 1 chegada por minuto, não por hora.",
            )

            modo_servico_p = st.radio(
                "Serviço (igual p/ todas as classes)",
                [
                    "Taxa μ (por unidade de tempo)",
                    "Tempo médio E[S] → μ = 1/E[S]",
                ],
                key="modo_servico_p",
                help="Mesma conversão que nos demais modelos; cada λₖ usa a unidade acima.",
            )
            if modo_servico_p.startswith("Taxa μ"):
                unidade_mu_p = st.selectbox(
                    "Unidade da taxa μ",
                    ["s", "min", "h", "dias"],
                    format_func=lambda u: RATE_UNIT_LABEL[u],
                    index=["s", "min", "h", "dias"].index(st.session_state.get("unidade_lam_p", "min")),
                    key="unidade_mu_p",
                )
                mu_p = st.number_input(
                    f"Taxa de serviço μ ({RATE_UNIT_LABEL[unidade_mu_p]})",
                    min_value=0.0001,
                    value=3.0,
                    step=0.0001,
                    format="%.4f",
                    key="mu_p_taxa",
                )
                mu_p_sec = rate_per_second(mu_p, unidade_mu_p)
            else:
                unidade_es_p = st.selectbox(
                    "Unidade de E[S]",
                    ["s", "min", "h", "dias"],
                    format_func=lambda u: DURATION_UNIT_LABEL[u],
                    index=1,
                    key="unidade_es_p",
                )
                es_p = st.number_input(
                    f"Tempo médio E[S] ({DURATION_UNIT_LABEL[unidade_es_p]})",
                    min_value=0.0001,
                    value=1 / 3.0,
                    step=0.0001,
                    format="%.4f",
                    key="mu_p_tempo",
                )
                mu_p_sec = mu_per_second_from_mean_service(es_p, unidade_es_p)
                st.caption(
                    f"μ equivalente ≈ **{rate_in_period(mu_p_sec, unidade_lam_p):.6f}** "
                    f"({RATE_UNIT_LABEL[unidade_lam_p]})"
                )

            s_p = st.number_input("Nº de servidores (s)", min_value=1, value=1, step=1)
            lambdas = []
            for k in range(int(num_classes)):
                lk = st.number_input(
                    f"λ{k+1} ({RATE_UNIT_LABEL[unidade_lam_p]} · prioridade {k+1})",
                    min_value=0.0001,
                    value=DEFAULT_PRIORITY_LAMBDAS[k],
                    step=0.0001,
                    format="%.4f",
                    key=f"lk{k}",
                )
                lambdas.append(lk)

            lambdas_sec = [rate_per_second(lk, unidade_lam_p) for lk in lambdas]

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
            if not is_priority:
                if lam_sec is None or mu_sec is None:
                    st.error(
                        "Não foi possível determinar λ e μ. Ajuste na barra lateral: chegadas "
                        "(taxa λ ou tempo entre chegadas), serviço (taxa μ ou tempo de serviço; "
                        "tempos podem ser simples ou compostos na barra lateral) "
                        "e confira os avisos na barra lateral."
                    )
                else:
                    title_rates = RATE_UNIT_LABEL[time_unit]
                    rho_simple = lam_sec / mu_sec if mu_sec and mu_sec > 0 else float("inf")

                    if model_key == "M/M/1":
                        res_raw = mm1(lam_sec, mu_sec)
                        if res_raw is None:
                            st.error(f"ρ = {rho_simple:.3f} ≥ 1. Sistema instável.")
                        else:
                            res = scale_result_times_for_display(res_raw, time_unit)
                            show_results(
                                res,
                                f"M/M/1  λ={lam_title:.4f} μ={mu_title:.4f} ({title_rates})",
                                time_unit,
                            )

                    elif model_key == "M/M/s":
                        s_v = int(s)
                        res_raw = mms(lam_sec, mu_sec, s_v)
                        rho_s = lam_sec / (s_v * mu_sec) if mu_sec else float("inf")
                        if res_raw is None:
                            st.error(f"ρ = {rho_s:.3f} ≥ 1. Sistema instável.")
                        else:
                            res = scale_result_times_for_display(res_raw, time_unit)
                            show_results(
                                res,
                                f"M/M/{s_v}  λ={lam_title:.4f} μ={mu_title:.4f} s={s_v} ({title_rates})",
                                time_unit,
                            )
                            st.markdown(
                                f"""<div class="info-box">
                        <b>Erlang-C (C)</b> = probabilidade de espera = {res_raw['C']:.5f}
                    </div>""",
                                unsafe_allow_html=True,
                            )

                    elif model_key == "M/M/1/K":
                        res_raw = mm1k(lam_sec, mu_sec, int(K))
                        res = scale_result_times_for_display(res_raw, time_unit)
                        show_results(
                            res,
                            f"M/M/1/K  λ={lam_title:.4f} μ={mu_title:.4f} K={int(K)} ({title_rates})",
                            time_unit,
                        )

                    elif model_key == "M/M/s/K":
                        s_v, K_v = int(s), int(K)
                        if K_v < s_v:
                            st.error("K deve ser ≥ s.")
                        else:
                            res_raw = mmsk(lam_sec, mu_sec, s_v, K_v)
                            res = scale_result_times_for_display(res_raw, time_unit)
                            st.markdown(
                                _html_detailed_input_conference(input_audit, lam_sec, mu_sec),
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                _html_rho_review(model_key, lam_sec, mu_sec, s_v),
                                unsafe_allow_html=True,
                            )
                            show_results(
                                res,
                                f"M/M/{s_v}/K  λ={lam_title:.4f} μ={mu_title:.4f} s={s_v} K={K_v} ({title_rates})",
                                time_unit,
                            )
                            _render_diagnosis_expander(res_raw, res, time_unit)

                    elif model_key == "M/M/1/N":
                        res_raw = mm1n(lam_sec, mu_sec, int(N))
                        res = scale_result_times_for_display(res_raw, time_unit)
                        st.markdown(
                            _html_detailed_input_conference(input_audit, lam_sec, mu_sec),
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            _html_rho_review(model_key, lam_sec, mu_sec, int(s)),
                            unsafe_allow_html=True,
                        )
                        show_results(
                            res,
                            f"M/M/1/N  λ={lam_title:.4f} μ={mu_title:.4f} N={int(N)} ({title_rates})",
                            time_unit,
                            N=int(N),
                            finite_model="mm1n",
                        )
                        _render_diagnosis_expander(res_raw, res, time_unit)

                    elif model_key == "M/M/s/N":
                        s_v, N_v = int(s), int(N)
                        res_raw = mmsn(lam_sec, mu_sec, s_v, N_v)
                        res = scale_result_times_for_display(res_raw, time_unit)
                        st.markdown(
                            _html_detailed_input_conference(input_audit, lam_sec, mu_sec),
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            _html_rho_review(model_key, lam_sec, mu_sec, s_v),
                            unsafe_allow_html=True,
                        )
                        show_results(
                            res,
                            f"M/M/{s_v}/N  λ={lam_title:.4f} μ={mu_title:.4f} s={s_v} N={N_v} ({title_rates})",
                            time_unit,
                            N=N_v,
                            finite_model="mmsn",
                            s_servers=s_v,
                        )
                        _render_diagnosis_expander(res_raw, res, time_unit)

                    elif is_mg1:
                        sigma2_sec = variance_to_seconds_squared(sigma2, unidade_sigma2)
                        res_raw = mg1(lam_sec, mu_sec, sigma2_sec)
                        rho_mg = lam_sec / mu_sec if mu_sec else float("inf")
                        if res_raw is None:
                            st.error(f"ρ = {rho_mg:.3f} ≥ 1. Sistema instável.")
                        else:
                            res = scale_result_times_for_display(res_raw, time_unit)
                            sigma2_disp_tu = sigma2_sec / (SECONDS_PER[time_unit] ** 2)
                            sig2_mm1_sec = 1 / (mu_sec**2)

                            st.markdown(
                                f'<div class="section-title">📊 M/G/1 — λ={lam_title:.4f} μ={mu_title:.4f} '
                                f"σ²={sigma2:.5f} ({DURATION_UNIT_LABEL[unidade_sigma2]})² · "
                                f"equiv. {sigma2_disp_tu:.5f} ({time_unit})²</div>",
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
                            res_mm1_raw = mg1(lam_sec, mu_sec, sig2_mm1_sec)
                            res_mm1 = scale_result_times_for_display(res_mm1_raw, time_unit)
                            if res_mm1:
                                ratio = res['Lq'] / res_mm1['Lq'] if res_mm1['Lq'] > 0 else float("nan")
                                cols = st.columns(3)
                                cols[0].metric("Lq (M/G/1)", f"{res['Lq']:.4f}")
                                cols[1].metric("Lq (M/M/1)", f"{res_mm1['Lq']:.4f}")
                                cols[2].metric("Razão Lq", f"{ratio:.4f}", help="Lq(M/G/1) / Lq(M/M/1)")

                                tu2 = SECONDS_PER[time_unit] ** 2
                                sig_vals_sec = np.linspace(0, 2 / mu_sec**2, 60)
                                sig_plot = sig_vals_sec / tu2
                                rho = res_raw["rho"]
                                lqs = [
                                    (lam_sec**2 * sv + rho**2) / (2 * (1 - rho))
                                    for sv in sig_vals_sec
                                ]
                                fig_g, ax = plt.subplots(figsize=(8, 3))
                                ax.plot(sig_plot, lqs, color="#e060b0", lw=2.5)
                                ax.axvline(
                                    sigma2_disp_tu,
                                    color="#a050c8",
                                    ls="--",
                                    lw=1.5,
                                    label=f"σ² atual ({time_unit}²)",
                                )
                                ax.axvline(
                                    sig2_mm1_sec / tu2,
                                    color="#dba8ef",
                                    ls="--",
                                    lw=1.5,
                                    label=f"σ²=1/μ² Exp ({time_unit}²)",
                                )
                                ax.set_xlabel(f"σ² em ({time_unit})²")
                                ax.set_ylabel("Lq")
                                ax.set_title("Sensibilidade de Lq à variância do serviço")
                                ax.legend()
                                ax.grid(alpha=0.3)
                                ax.spines["top"].set_visible(False)
                                ax.spines["right"].set_visible(False)
                                fig_g.patch.set_facecolor("#f7f9fc")
                                ax.set_facecolor("#f7f9fc")
                                plt.tight_layout()
                                st.pyplot(fig_g, use_container_width=True)

            elif is_priority:
                _mn = unicodedata.normalize("NFC", model.strip())
                preemptive = "Sem interrupção" not in _mn
                lam_tot = sum(lambdas)
                lam_tot_sec = sum(lambdas_sec)
                rho_tot = lam_tot_sec / (int(s_p) * mu_p_sec) if mu_p_sec > 0 else float("inf")
                mu_display = rate_in_period(mu_p_sec, time_unit)

                if rho_tot >= 1:
                    st.error(f"ρ total = {rho_tot:.3f} ≥ 1. Sistema instável.")
                else:
                    results = (
                        priority_preemptive(lambdas_sec, mu_p_sec, int(s_p))
                        if preemptive
                        else priority_nonpreemptive(lambdas_sec, mu_p_sec, int(s_p))
                    )
                    for i, r in enumerate(results):
                        rr = scale_result_times_for_display(r, time_unit)
                        r["W"], r["Wq"] = rr["W"], rr["Wq"]
                        r["lam"] = lambdas[i]

                    tipo = "Com interrupção (Preemptivo)" if preemptive else "Sem interrupção (Não-preemptivo)"
                    st.markdown(
                        f'<div class="section-title">📊 Prioridades — {tipo} | s={int(s_p)} '
                        f"μ={mu_display:.4f} ({RATE_UNIT_LABEL[time_unit]})</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                <div class="info-box">
                  λ total = {lam_tot:.3f} ({RATE_UNIT_LABEL[unidade_lam_p]}) · μ = {mu_display:.4f} ({RATE_UNIT_LABEL[time_unit]}) · s = {int(s_p)} · ρ total = {rho_tot:.4f}
                </div>""",
                        unsafe_allow_html=True,
                    )
                    show_priority_results(results, time_unit)
                    st.pyplot(plot_priority_bars(results, time_unit), use_container_width=True)

                    st.markdown("---")
                    df_prio = pd.DataFrame(
                        [
                            {
                                "Prioridade": f"P{r['k']}",
                                "λₖ": r["lam"],
                                "ρₖ": f"{r['rho']:.4f}",
                                "W": f"{r['W']:.5f}",
                                "Wq": f"{r['Wq']:.5f}",
                                "L": f"{r['L']:.5f}",
                                "Lq": f"{r['Lq']:.5f}",
                            }
                            for r in results
                        ]
                    )
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
        st.caption(
            "Nesta aba, **λ e μ devem estar na mesma unidade de taxa** (ex.: ambos «por minuto»). "
            "Use a barra lateral para misturar unidades (ex.: λ/h e E[S] em minutos)."
        )
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