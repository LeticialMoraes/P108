import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def plot_probs(Pn, title="Distribuição P(n)"):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ns = list(range(len(Pn)))
    cols = ['#e060b0' if i == 0 else '#b07fd4' for i in ns]
    bars = ax.bar(ns, Pn, color=cols, edgecolor='white', linewidth=0.8, zorder=3)
    ax.set_xlabel("n (clientes)", fontsize=9)
    ax.set_ylabel("P(n)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.set_xticks(ns)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor('#f7f9fc')
    ax.set_facecolor('#f7f9fc')
    for bar, v in zip(bars, Pn):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + .001,
            f'{v:.3f}',
            ha='center',
            va='bottom',
            fontsize=7,
            fontfamily='monospace',
        )
    plt.tight_layout()
    return fig


def metrics_html(items):
    html = '<div class="metric-grid">'
    for label, val, desc in items:
        html += f"""<div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
            <div class="unit">{desc}</div>
        </div>"""
    return html + '</div>'


def show_results(res, title, time_unit="min", N=None, finite_model=None, s_servers=1):
    st.markdown(f'<div class="section-title">📊 {title}</div>', unsafe_allow_html=True)
    items = [
        ("P₀", f"{res['P0']:.5f}", "prob. sistema vazio"),
        ("ρ", f"{res['rho']:.4f}", "utilização"),
    ]
    if 'lam_eff' in res:
        items.append(("λ̄", f"{res['lam_eff']:.4f}", f"taxa efetiva ({time_unit}⁻¹)"))
    if 'PK' in res:
        items.append(("P_K", f"{res['PK']:.5f}", "prob. bloqueio"))
    items += [
        ("L", f"{res['L']:.4f}", "clientes no sistema"),
        ("Lq", f"{res['Lq']:.4f}", "clientes na fila"),
        ("W", f"{res['W']:.4f}", f"tempo no sistema ({time_unit})"),
        ("Wq", f"{res['Wq']:.4f}", f"espera na fila ({time_unit})"),
    ]
    if N is not None and finite_model is None:
        ativos = N - res["L"]
        items.append(("N − L", f"{ativos:.4f}", "população ativa (operacional)"))
    st.markdown(metrics_html(items), unsafe_allow_html=True)

    if finite_model == "mm1n" and N is not None:
        L = float(res["L"])
        p0 = float(res["P0"])
        util_srv = 1.0 - p0
        n_minus_l = float(N) - L
        prop = n_minus_l / float(N) if N else float("nan")
        op_items = [
            ("1 − P₀", f"{util_srv:.5f}", "utilização do servidor"),
            ("(N − L)/N", f"{prop:.5f}", "prop. média da pop. operando"),
            ("N − L", f"{n_minus_l:.4f}", "qtd. média da pop. operando"),
        ]
        st.markdown(
            '<div class="section-title" style="margin-top:1.1rem">🔧 Indicadores operacionais</div>',
            unsafe_allow_html=True,
        )
        st.markdown(metrics_html(op_items), unsafe_allow_html=True)

    elif finite_model == "mmsn" and N is not None:
        L = float(res["L"])
        n_minus_l = float(N) - L
        prop = n_minus_l / float(N) if N else float("nan")
        rho_bar = float(res["rho"])
        s_v = int(s_servers)
        op_items = [
            ("N − L", f"{n_minus_l:.4f}", "qtd. média da pop. operando"),
            ("(N − L)/N", f"{prop:.5f}", "prop. média da pop. operando"),
            ("λ̄/(sμ)", f"{rho_bar:.5f}", f"util. média dos servidores (s = {s_v})"),
        ]
        st.markdown(
            '<div class="section-title" style="margin-top:1.1rem">🔧 Indicadores operacionais</div>',
            unsafe_allow_html=True,
        )
        st.markdown(metrics_html(op_items), unsafe_allow_html=True)
    if 'Pn' in res:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<div class="section-title">Tabela P(n)</div>', unsafe_allow_html=True)
            df = pd.DataFrame({
                "n": range(len(res['Pn'])),
                "P(n)": [f"{p:.6f}" for p in res['Pn']],
                "%": [f"{p*100:.3f}%" for p in res['Pn']],
            })
            st.dataframe(df, use_container_width=True, hide_index=True, height=280)
        with c2:
            st.markdown('<div class="section-title">Gráfico</div>', unsafe_allow_html=True)
            st.pyplot(plot_probs(res['Pn']), use_container_width=True)


def show_priority_results(results, time_unit="min"):
    for r in results:
        k = r['k']
        color = [
            "#e060b0",
            "#b07fd4",
            "#dba8ef",
            "#c084dc",
            "#a050c8",
        ][(k - 1) % 5]
        st.markdown(
            f"""
        <div class="priority-card">
          <h4><span class="prio-badge">Prioridade {k}</span> λ={r['lam']} · ρ={r['rho']:.4f}</h4>
        </div>""",
            unsafe_allow_html=True,
        )
        items = [
            ("W", f"{r['W']:.5f}", f"tempo no sistema ({time_unit})"),
            ("Wq", f"{r['Wq']:.5f}", f"espera na fila ({time_unit})"),
            ("L", f"{r['L']:.5f}", "clientes no sistema"),
            ("Lq", f"{r['Lq']:.5f}", "clientes na fila"),
        ]
        st.markdown(metrics_html(items), unsafe_allow_html=True)


def plot_priority_bars(results, time_unit):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    ks = [f"P{r['k']}" for r in results]
    Ws = [r['W'] for r in results]
    Wqs = [r['Wq'] for r in results]
    Ls = [r['L'] for r in results]
    Lqs = [r['Lq'] for r in results]
    clrs = ['#e060b0', '#b07fd4', '#dba8ef', '#c084dc', '#a050c8'][:len(results)]
    x = range(len(ks))
    axes[0].bar([i - .2 for i in x], Ws, .35, label=f'W ({time_unit})', color=clrs, alpha=.9)
    axes[0].bar([i + .2 for i in x], Wqs, .35, label=f'Wq ({time_unit})', color=clrs, alpha=.5, hatch='//')
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(ks)
    axes[0].set_title("Tempos médios por prioridade")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', alpha=.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].set_facecolor('#f7f9fc')

    axes[1].bar([i - .2 for i in x], Ls, .35, label='L', color=clrs, alpha=.9)
    axes[1].bar([i + .2 for i in x], Lqs, .35, label='Lq', color=clrs, alpha=.5, hatch='//')
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(ks)
    axes[1].set_title("Clientes médios por prioridade")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].set_facecolor('#f7f9fc')
    fig.patch.set_facecolor('#f7f9fc')
    plt.tight_layout()
    return fig
