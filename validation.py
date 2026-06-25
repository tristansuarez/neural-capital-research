"""
Motor de validacion del laboratorio. Aqui vive el rigor.

Tres piezas:
  1) walk_forward(): evalua el modelo SIEMPRE fuera de muestra. En cada dia el
     modelo solo ve el pasado; se re-entrena periodicamente. Imposible mirar el
     futuro por construccion.
  2) Metricas serias: rendimiento neto de costes, Sharpe, maximo drawdown,
     acierto. El Sharpe es la metrica central porque es ajustada al riesgo.
  3) Significancia por bootstrap de bloques: ¿el Sharpe es distinguible del
     azar? Si pruebas muchos modelos, alguno parecera bueno por suerte. Esto es
     lo que casi nadie hace y lo que separa el laboratorio de un canal de humo.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

ANN = 252  # dias de mercado al año


def walk_forward(df: pd.DataFrame, model, train_window: int,
                 refit_every: int) -> pd.DataFrame:
    """Devuelve los pesos OUT-OF-SAMPLE por activo y dia (sin lookahead)."""
    assets = model.assets
    rows, idx = [], []
    last_fit = -10**9
    for i in range(train_window, len(df)):
        hist = df.iloc[:i]                 # datos hasta el dia i-1 incluido
        if i - last_fit >= refit_every:
            model.fit(hist)               # re-entrenamiento solo con el pasado
            last_fit = i
        w = model.weights(hist)           # pesos para el dia i
        rows.append([w.get(a, 0.0) for a in assets])
        idx.append(df.index[i])
    return pd.DataFrame(rows, index=idx, columns=assets)


def strategy_returns(weights: pd.DataFrame, prices: pd.DataFrame,
                     cost_bps: float = 2.0) -> pd.Series:
    """Rendimiento diario neto de costes de transaccion."""
    assets = list(weights.columns)
    asset_ret = prices[assets].pct_change().reindex(weights.index)
    gross = (weights * asset_ret).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    cost = turnover * (cost_bps / 1e4)
    return (gross - cost).fillna(0.0)


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def metrics(net_ret: pd.Series, weights: pd.DataFrame | None = None) -> dict:
    r = net_ret.dropna()
    n = len(r)
    if n == 0 or r.std(ddof=1) == 0:
        return {"n_dias": n, "sharpe": 0.0, "cagr": 0.0, "vol_anual": 0.0,
                "max_drawdown": 0.0, "acierto": 0.0, "rotaciones": 0}
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(ANN))
    equity = (1 + r).cumprod()
    cagr = float(equity.iloc[-1] ** (ANN / n) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(ANN))
    active = r[r != 0]
    hit = float((active > 0).mean()) if len(active) else 0.0
    rot = int((weights.diff().abs().sum(axis=1) > 1e-6).sum()) if weights is not None else 0
    return {"n_dias": n, "sharpe": round(sharpe, 3), "cagr": round(cagr, 4),
            "vol_anual": round(vol, 4), "max_drawdown": round(_max_drawdown(equity), 4),
            "acierto": round(hit, 4), "rotaciones": rot}


def bootstrap_sharpe_pvalue(net_ret: pd.Series, n_boot: int = 2000,
                            block: int = 20, seed: int = 7) -> dict:
    """
    Bootstrap de bloques circular. Devuelve el p-valor (una cola) de que el
    Sharpe sea <= 0 y un intervalo de confianza al 90%. Bloques para respetar
    la autocorrelacion de los rendimientos.
    """
    r = net_ret.dropna().values
    n = len(r)
    if n < block * 2 or r.std(ddof=1) == 0:
        return {"p_valor": 1.0, "ic90": [0.0, 0.0]}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        sample = r[idx][:n]
        sd = sample.std(ddof=1)
        sharpes[b] = sample.mean() / sd * np.sqrt(ANN) if sd > 0 else 0.0
    p = float(np.mean(sharpes <= 0))
    lo, hi = np.percentile(sharpes, [5, 95])
    return {"p_valor": round(p, 4), "ic90": [round(float(lo), 3), round(float(hi), 3)]}


def evaluate(df: pd.DataFrame, model, train_window: int, refit_every: int,
             cost_bps: float = 2.0, equity_points: int = 300) -> dict:
    """Ejecuta el walk-forward completo y devuelve el informe del modelo."""
    W = walk_forward(df, model, train_window, refit_every)
    net = strategy_returns(W, df, cost_bps)
    m = metrics(net, W)
    sig = bootstrap_sharpe_pvalue(net)

    equity = (1 + net).cumprod()
    step = max(1, len(equity) // equity_points)
    curve = equity.iloc[::step]
    curva = [{"fecha": d.strftime("%Y-%m-%d"), "valor": round(float(v), 4)}
             for d, v in curve.items()]

    extras = {}
    for k in ("cointegrated", "coint_pvalue", "half_life", "beta"):
        if hasattr(model, k) and getattr(model, k) is not None:
            v = getattr(model, k)
            if isinstance(v, (bool, np.bool_)):
                extras[k] = bool(v)
            elif isinstance(v, (int, float, np.integer, np.floating)):
                extras[k] = round(float(v), 4)
            else:
                extras[k] = v

    return {
        "modelo": model.name,
        "descripcion": model.description,
        "metricas": m,
        "significancia": sig,
        "diagnostico": extras,
        "curva": curva,
    }
