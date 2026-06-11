import math


def fac(n):
    return math.factorial(int(n))


def mm1(lam, mu):
    if mu <= 0:
        return None

    if lam < 0:
        return None

    rho = lam / mu

    if rho >= 1:
        return None

    P0 = 1 - rho

    Lq = rho**2 / (1 - rho)
    L = rho + Lq

    Wq = Lq / lam
    W = Wq + (1 / mu)

    # Probabilidades extras
    P_Wq_1 = rho * math.exp(-(mu - lam))
    P_W_1 = math.exp(-(mu - lam))

    return dict(
        P0=P0,
        rho=rho,
        L=L,
        Lq=Lq,
        W=W,
        Wq=Wq,
        P_Wq_1=P_Wq_1,
        P_W_1=P_W_1
    )


def mms(lam, mu, s):
    if mu <= 0:
        return None
    
    if lam < 0:
        return None
    
    if s < 1:
        return None

    rho = lam / (s * mu)
    r = lam / mu
    if rho >= 1:
        return None
    sum_terms = sum(r**n / fac(n) for n in range(s))
    last = r**s / (fac(s) * (1 - rho))
    P0 = 1 / (sum_terms + last)
    C = last * P0
    Lq = C * rho / (1 - rho)
    L = Lq + r
    Wq = Lq / lam
    W = Wq + 1 / mu
    return dict(P0=P0, rho=rho, C=C, L=L, Lq=Lq, W=W, Wq=Wq)


def mm1k(lam, mu, K):
    if mu <= 0:
        return None
    
    if lam < 0:
        return None
    
    if K < 0:
        return None

    rho = lam / mu
    if abs(rho - 1.0) < 1e-12:
        P0 = 1 / (K + 1)
        Pn = [P0] * (K + 1)
    else:
        P0 = (1 - rho) / (1 - rho**(K + 1))
        Pn = [P0 * rho**n for n in range(K + 1)]
    PK = Pn[K]
    lam_eff = lam * (1 - PK)
    L = rho / (1 - rho) - (K + 1) * rho**(K + 1) / (1 - rho**(K + 1)) if abs(rho - 1) > 1e-12 else K / 2
    Lq = L - (1 - P0)
    W = L / lam_eff if lam_eff > 0 else float('inf')
    Wq = max(Lq, 0) / lam_eff if lam_eff > 0 else float('inf')
    return dict(P0=P0, Pn=Pn, PK=PK, rho=rho, lam_eff=lam_eff, L=L, Lq=max(Lq, 0), W=W, Wq=Wq)


def mmsk(lam, mu, s, K):
    rho = lam / (s * mu)
    r = lam / mu
    s1 = sum(r**n / fac(n) for n in range(s))
    s2 = sum(r**n / (fac(s) * s**(n - s)) for n in range(s, K + 1))
    P0 = 1 / (s1 + s2)
    Pn = [(r**n / fac(n) * P0 if n <= s else r**n / (fac(s) * s**(n - s)) * P0) for n in range(K + 1)]
    PK = Pn[K]
    lam_eff = lam * (1 - PK)
    if abs(1 - rho) < 1e-12:
        Lq = P0 * r**s / fac(s) * (K - s) * (K - s + 1) / 2
    else:
        Lq = (P0 * r**s * rho) / (fac(s) * (1 - rho)**2) * (
            1 - rho**(K - s) - (K - s) * rho**(K - s) * (1 - rho)
        )
    sp = sum(Pn[n] for n in range(s))
    L = sum(n * Pn[n] for n in range(s)) + Lq + s * (1 - sp)
    W = L / lam_eff if lam_eff > 0 else float('inf')
    Wq = max(Lq, 0) / lam_eff if lam_eff > 0 else float('inf')
    return dict(P0=P0, Pn=Pn, PK=PK, rho=rho, lam_eff=lam_eff, L=L, Lq=max(Lq, 0), W=W, Wq=Wq)


def mm1n(lam, mu, N):
    Pn = [0.0] * (N + 1)
    # Probabilidade não normalizada
    Pn[0] = 1.0
    for n in range(1, N + 1):
        lambda_prev = (N - (n - 1)) * lam
        # Como é M/M/1/N:
        mu_curr = mu
        Pn[n] = Pn[n - 1] * lambda_prev / mu_curr

    # Normalização
    norm = sum(Pn)
    P0 = 1 / norm
    Pn = [p * P0 for p in Pn]
    # Número médio no sistema
    L = sum(n * p for n, p in enumerate(Pn))
    # Taxa efetiva
    lam_eff = sum(
        (N - n) * lam * Pn[n]
        for n in range(N + 1)
    )
    # Número médio em serviço
    Ls = 1 - P0
    # Número médio na fila
    Lq = max(L - Ls, 0)
    # Tempos médios
    W = L / lam_eff if lam_eff > 0 else float("inf")
    Wq = Lq / lam_eff if lam_eff > 0 else float("inf")
    # Utilização real
    rho = lam_eff / mu
    return dict(
        P0=P0,
        Pn=Pn,
        rho=rho,
        lam_eff=lam_eff,
        L=L,
        Lq=Lq,
        W=W,
        Wq=Wq,
    )


def mmsn(lam, mu, s, N):
    Pn = [0.0] * (N + 1)
    Pn[0] = 1.0
    for n in range(1, N + 1):
        lambda_prev = (N - (n - 1)) * lam
        mu_curr = min(n, s) * mu
        Pn[n] = Pn[n - 1] * lambda_prev / mu_curr
    norm = sum(Pn)
    P0 = 1 / norm
    Pn = [p * P0 for p in Pn]
    L = sum(n * p for n, p in enumerate(Pn))
    lam_eff = sum(
        (N - n) * lam * Pn[n]
        for n in range(N + 1)
    )
    Lq = sum(
        max(n - s, 0) * Pn[n]
        for n in range(N + 1)
    )
    W = L / lam_eff if lam_eff > 0 else float("inf")
    Wq = Lq / lam_eff if lam_eff > 0 else float("inf")
    rho = lam_eff / (s * mu)
    return dict(
        P0=P0,
        Pn=Pn,
        rho=rho,
        lam_eff=lam_eff,
        L=L,
        Lq=Lq,
        W=W,
        Wq=Wq,
    )


# =========================
# M/G/1
# Pollaczek–Khinchine
# =========================
def mg1(lam, mu, sigma2=None):

    rho = lam/mu

    if rho >= 1:
        return None

    P0 = 1-rho

    # Se não passar variância:
    # assume exponencial

    if sigma2 is None:
        sigma2 = (1/mu)**2

    Lq = (
        (lam**2*sigma2 + rho**2)
        /
        (2*(1-rho))
    )

    L = rho + Lq

    Wq = Lq/lam

    W = Wq + (1/mu)

    return {
        "P0":P0,
        "rho":rho,
        "sigma2":sigma2,
        "L":L,
        "Lq":Lq,
        "W":W,
        "Wq":Wq
    }

def priority_preemptive(lambdas, mu, s=1):
    K = len(lambdas)

    rho = [l / (s * mu) for l in lambdas]

    cumrho = [0.0]
    for r in rho:
        cumrho.append(cumrho[-1] + r)

    results = []

    for k in range(K):
        prev = cumrho[k]
        curr = cumrho[k + 1]

        lk = lambdas[k]

        if abs(1 - prev) < 1e-12 or abs(1 - curr) < 1e-12:
            Wk = float('inf')
        else:
            Wk = (1 / mu) / ((1 - prev) * (1 - curr))

        Wqk = Wk - 1 / mu

        # igual ao gabarito
        lam_acum = sum(lambdas[:k+1])

        Lk = lam_acum * Wk
        Lqk = Lk - (lam_acum / mu)

        results.append(
            dict(
                k=k + 1,
                lam=lk,
                rho=rho[k],
                W=Wk,
                Wq=max(Wqk, 0),
                L=Lk,
                Lq=max(Lqk, 0)
            )
        )

    return results


def priority_nonpreemptive(lambdas, mu, s=1):
    K = len(lambdas)
    lam_total = sum(lambdas)
    r = lam_total / mu
    rho_i = [l / (s * mu) for l in lambdas]
    cumrho = [0.0]
    for ri in rho_i:
        cumrho.append(cumrho[-1] + ri)
    erlang_sum = sum(r**j / fac(j) for j in range(s))
    denom_base = fac(s) * (s * mu - lam_total) / r**s * erlang_sum + s * mu
    results = []
    for k in range(K):
        prev = cumrho[k]
        curr = cumrho[k + 1]
        lk = lambdas[k]
        if abs(1 - prev) < 1e-12 or abs(1 - curr) < 1e-12 or denom_base <= 0:
            Wk = float('inf')
        else:
            Wk = 1 / (denom_base * (1 - prev) * (1 - curr)) + 1 / mu
        Wqk = Wk - 1 / mu
        Lk = lk * Wk
        Lqk = lk * Wqk
        results.append(dict(k=k + 1, lam=lk, rho=rho_i[k], W=Wk, Wq=max(Wqk, 0), L=Lk, Lq=max(Lqk, 0)))
    return results


__all__ = [
    'fac', 'mm1', 'mms', 'mm1k', 'mmsk', 'mm1n', 'mmsn', 'mg1',
    'priority_preemptive', 'priority_nonpreemptive',
]
