import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             confusion_matrix, classification_report, f1_score,
                             precision_score, recall_score)

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

# Paleta e estilo dos gráficos
AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({
    "figure.figsize": (10, 4), "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.edgecolor": "#b9b8b2", "xtick.color": "#52514e", "ytick.color": "#52514e",
})
pct = mtick.PercentFormatter(1.0, decimals=0)

ARQ_HIST = "transacoes-hist-nivea.xlsx"     # histórico 2025 com rótulo
hist = pd.read_excel(ARQ_HIST, sheet_name=0)
jan  = pd.read_excel(ARQ_JAN,  sheet_name=0)

print(f"Histórico : {hist.shape[0]:,} linhas x {hist.shape[1]} colunas")
print(f"Janeiro/26: {jan.shape[0]:,} linhas x {jan.shape[1]} colunas")
print("\nColunas do histórico:", list(hist.columns))
def corrigir_texto(s):
    """Corrige textos gravados em UTF-8 e lidos como Latin-1 (ex.: 'SÃ£o Paulo')."""
    try:
        return s.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError, AttributeError):
        return s

def preparar(df, col_data):
    df = df.copy()
    # padroniza nomes de colunas
    cols = list(df.columns)
    cols = ["tipo_estabelecimento" if (not isinstance(c, str)) else c for c in cols]
    df.columns = cols
    df = df.rename(columns={col_data: "data", "cliente Nº": "cliente_id"})
    # textos
    for c in df.select_dtypes(include=["object", "string"]).columns:
        if c != "HORARIO":
            df[c] = df[c].map(corrigir_texto)
    # data e hora
    hora = pd.to_datetime(df["HORARIO"].astype(str), format="%H:%M:%S")
    df["hora"] = hora.dt.hour
    df["minuto"] = hora.dt.minute
    df["datahora"] = pd.to_datetime(df["data"].dt.strftime("%Y-%m-%d") + " " + df["HORARIO"].astype(str))
    df["dia_semana"] = df["data"].dt.dayofweek   # 0 = segunda
    df["dia_mes"] = df["data"].dt.day
    df["mes"] = df["data"].dt.month
    return df

hist = preparar(hist, "data")
jan  = preparar(jan, "Data")
hist["fraude"] = hist["flag_fraude"].astype(int)

print("Valores nulos (histórico):", int(hist.isna().sum().sum()))
print("Valores nulos (janeiro)  :", int(jan.isna().sum().sum()))
print("Linhas duplicadas (histórico):", int(hist.duplicated().sum()))
print("Período histórico:", hist["data"].min().date(), "a", hist["data"].max().date())
print("Período janeiro  :", jan["data"].min().date(), "a", jan["data"].max().date())
print("\nCidades:", sorted(hist["cidade"].unique()))
# Resumo das colunas
resumo = pd.DataFrame({
    "tipo": hist.dtypes.astype(str),
    "valores_distintos": hist.nunique(),
    "exemplo": hist.iloc[0],
})
resumo.loc[["data","HORARIO","nome_estabelecimento","tipo_estabelecimento","valor_transacao","cliente_id",
            "cidade","device_type","is_new_device","temperatura_celsius","codigo_postal","score_interno",
            "turno","valor_categoria","cidade_regiao","flag_fraude"]]
            # Os mesmos estabelecimentos, cidades e clientes aparecem nas duas bases?
for c in ["nome_estabelecimento", "tipo_estabelecimento", "cidade", "device_type", "turno", "valor_categoria"]:
    novos = set(jan[c]) - set(hist[c])
    print(f"{c:22s} categorias novas em jan/26: {novos or 'nenhuma'}")
print(f"Clientes de jan/26 presentes no histórico: {jan['cliente_id'].isin(hist['cliente_id']).mean():.1%}")
print(f"codigo_postal distintos no histórico: {hist['codigo_postal'].nunique():,} (praticamente um por transação → não será usado)")
taxa = hist["fraude"].mean()
cont = hist["fraude"].value_counts().rename({0: "Não fraudulenta", 1: "Fraudulenta"})
print(cont.to_string())
print(f"\nTaxa de fraude no histórico: {taxa:.2%}  (≈ 1 fraude a cada {1/taxa:.0f} transações)")
por_hora = hist.groupby("hora")["fraude"].mean()

fig, ax = plt.subplots(figsize=(11, 4))
cores = [LARANJA if v > 2 * taxa else AZUL for v in por_hora.values]
ax.bar(por_hora.index, por_hora.values, color=cores, width=0.8)
ax.axhline(taxa, color=CINZA, ls="--", lw=1)
ax.text(23.6, taxa, f" média {taxa:.1%}", va="bottom", ha="right", color="#52514e", fontsize=9)
for h in [12, 13, 14, 22, 23]:
    ax.text(h, por_hora[h] + 0.005, f"{por_hora[h]:.0%}", ha="center", fontsize=9)
ax.set_xticks(range(24)); ax.yaxis.set_major_formatter(pct)
ax.set_xlabel("Hora da transação"); ax.set_ylabel("Taxa de fraude")
ax.set_title("Taxa de fraude por hora do dia (laranja = mais que o dobro da média)")
plt.show()
cats = [("tipo_estabelecimento", "Tipo de estabelecimento"), ("is_new_device", "Dispositivo novo?"),
        ("device_type", "Tipo de dispositivo"), ("cidade", "Cidade"),
        ("turno", "Turno"), ("valor_categoria", "Faixa de valor")]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (col, titulo) in zip(axes.ravel(), cats):
    s = hist.groupby(col)["fraude"].mean().sort_values()
    ax.barh(s.index.astype(str), s.values, color=AZUL, height=0.6)
    ax.axvline(taxa, color=CINZA, ls="--", lw=1)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:.1%}", va="center", fontsize=9)
    ax.xaxis.set_major_formatter(pct); ax.set_title(titulo)
    ax.set_xlim(0, s.max() * 1.25); ax.grid(axis="y", visible=False)
fig.suptitle("Taxa de fraude por categoria (linha tracejada = média geral)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
leg, fr = hist[hist.fraude == 0], hist[hist.fraude == 1]

bins = np.logspace(np.log10(0.5), np.log10(hist.valor_transacao.max()), 50)
axes[0].hist(leg.valor_transacao, bins=bins, density=True, alpha=0.55, color=AZUL, label="Não fraudulenta")
axes[0].hist(fr.valor_transacao, bins=bins, density=True, alpha=0.55, color=LARANJA, label="Fraudulenta")
axes[0].set_xscale("log"); axes[0].set_title("Valor da transação (R$, escala log)")

for ax, col, titulo in [(axes[1], "score_interno", "Score interno"),
                        (axes[2], "temperatura_celsius", "Temperatura (°C)")]:
    b = np.linspace(hist[col].min(), hist[col].max(), 40)
    ax.hist(leg[col], bins=b, density=True, alpha=0.55, color=AZUL, label="Não fraudulenta")
    ax.hist(fr[col], bins=b, density=True, alpha=0.55, color=LARANJA, label="Fraudulenta")
    ax.set_title(titulo)
for ax in axes: ax.set_yticks([]); ax.set_ylabel("densidade")
axes[0].legend(frameon=False)
plt.tight_layout(); plt.show()

hist.groupby("flag_fraude")[["valor_transacao", "score_interno", "temperatura_celsius"]].median() \
    .rename(index={False: "Não fraudulenta", True: "Fraudulenta"}).rename_axis("mediana")
    piv = hist.pivot_table(index="tipo_estabelecimento", columns="hora", values="fraude", aggfunc="mean")
piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(13, 3.6))
im = ax.imshow(piv.values, aspect="auto", cmap="Blues", vmin=0)
ax.set_xticks(range(24)); ax.set_xticklabels(piv.columns)
ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
ax.grid(False); ax.set_xlabel("Hora")
ax.set_title("Taxa de fraude por hora e tipo de estabelecimento")
cb = plt.colorbar(im, ax=ax, format=pct, pad=0.01); cb.outline.set_visible(False)
plt.show()

piv[[12, 13, 14, 22, 23]].style.format("{:.1%}").background_gradient(cmap="Blues", axis=None)
mensal = hist.groupby("mes")["fraude"].mean()
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(mensal.index, mensal.values, color=AZUL, lw=2, marker="o", ms=6)
ax.text(12, mensal[12] + 0.001, f"{mensal[12]:.1%}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(1, 13)); ax.set_ylim(0, mensal.max() * 1.25); ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
ax.set_xlabel("Mês de 2025"); ax.set_ylabel("Taxa de fraude"); ax.set_title("Taxa de fraude mensal")
plt.show()
NUM = ["hora", "minuto", "dia_semana", "dia_mes", "valor_transacao", "is_new_device",
       "temperatura_celsius", "score_interno"]
CAT = ["nome_estabelecimento", "tipo_estabelecimento", "cidade", "device_type",
       "turno", "valor_categoria", "cidade_regiao"]

def matriz_features(df):
    X = df[NUM + CAT].copy()
    X["is_new_device"] = X["is_new_device"].astype(int)
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=sorted(hist[c].unique()))
    return X

X, y = matriz_features(hist), hist["fraude"]
treino = hist["data"] <  "2025-10-01"
valid  = (hist["data"] >= "2025-10-01") & (hist["data"] < "2025-11-01")
teste  = hist["data"] >= "2025-11-01"

pd.DataFrame({
    "linhas": [treino.sum(), valid.sum(), teste.sum()],
    "taxa_fraude": [y[treino].mean(), y[valid].mean(), y[teste].mean()],
}, index=["Treino (jan–set)", "Validação (out)", "Teste (nov–dez)"]).style.format({"linhas": "{:,}", "taxa_fraude": "{:.2%}"})
# Modelo 1 — Regressão Logística (hora tratada como categórica)
CAT_LOG = CAT + ["hora"]
NUM_LOG = [c for c in NUM if c != "hora"]
logreg = Pipeline([
    ("prep", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_LOG),
        ("num", StandardScaler(), NUM_LOG),
    ])),
    ("clf", LogisticRegression(max_iter=2000, C=1.0)),
])
X_log = X.copy(); X_log["hora"] = X_log["hora"].astype(str)
for c in CAT: X_log[c] = X_log[c].astype(str)
logreg.fit(X_log[treino], y[treino])

# Modelo 2 — Gradient Boosting
PARAMS_GB = dict(categorical_features="from_dtype", learning_rate=0.05, max_iter=400,
                 max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0, random_state=SEED)
gb = HistGradientBoostingClassifier(**PARAMS_GB)
gb.fit(X[treino], y[treino])

p_val = {"Regressão Logística": logreg.predict_proba(X_log[valid])[:, 1],
         "Gradient Boosting":   gb.predict_proba(X[valid])[:, 1]}

def melhor_limiar(y_true, prob):
    pr, rc, th = precision_recall_curve(y_true, prob)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    i = np.nanargmax(f1[:-1])
    return th[i], f1[i]

linhas = []
for nome, p in p_val.items():
    lim, f1v = melhor_limiar(y[valid], p)
    linhas.append({"modelo": nome, "ROC-AUC": roc_auc_score(y[valid], p),
                   "PR-AUC": average_precision_score(y[valid], p), "melhor F1": f1v, "limiar": lim})
comp_val = pd.DataFrame(linhas).set_index("modelo")
comp_val
LIMIAR = comp_val.loc["Gradient Boosting", "limiar"]
pv = p_val["Gradient Boosting"]
print(f"Limiar escolhido na validação (máximo F1): {LIMIAR:.3f}")
print(f"→ precisão {precision_score(y[valid], pv >= LIMIAR):.1%} | recall {recall_score(y[valid], pv >= LIMIAR):.1%}")

# Como precisão e recall mudam com o limiar
pr, rc, th = precision_recall_curve(y[valid], pv)
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(th, pr[:-1], color=AZUL, lw=2, label="Precisão")
ax.plot(th, rc[:-1], color=LARANJA, lw=2, label="Recall")
ax.axvline(LIMIAR, color=CINZA, ls="--", lw=1)
ax.text(LIMIAR, 0.03, f" limiar {LIMIAR:.2f}", color="#52514e", fontsize=9)
ax.set_xlabel("Limiar de probabilidade"); ax.yaxis.set_major_formatter(pct)
ax.set_title("Troca entre precisão e recall (validação, Gradient Boosting)"); ax.legend(frameon=False)
plt.show()
p_te_gb  = gb.predict_proba(X[teste])[:, 1]
p_te_log = logreg.predict_proba(X_log[teste])[:, 1]
pred_te = (p_te_gb >= LIMIAR).astype(int)

res_teste = pd.DataFrame({
    "ROC-AUC":  [roc_auc_score(y[teste], p_te_log), roc_auc_score(y[teste], p_te_gb)],
    "PR-AUC":   [average_precision_score(y[teste], p_te_log), average_precision_score(y[teste], p_te_gb)],
    "Precisão": [precision_score(y[teste], p_te_log >= comp_val.loc["Regressão Logística","limiar"]),
                 precision_score(y[teste], pred_te)],
    "Recall":   [recall_score(y[teste], p_te_log >= comp_val.loc["Regressão Logística","limiar"]),
                 recall_score(y[teste], pred_te)],
    "F1":       [f1_score(y[teste], p_te_log >= comp_val.loc["Regressão Logística","limiar"]),
                 f1_score(y[teste], pred_te)],
}, index=["Regressão Logística", "Gradient Boosting"])
print(f"Referência: PR-AUC de um modelo aleatório = taxa de fraude = {y[teste].mean():.3f}\n")
res_teste.style.format("{:.3f}").highlight_max(axis=0, props="font-weight:bold")
print(classification_report(y[teste], pred_te, target_names=["Não fraudulenta", "Fraudulenta"], digits=3))

cm = confusion_matrix(y[teste], pred_te)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
ax = axes[0]
ax.imshow(cm, cmap="Blues"); ax.grid(False)
rot = ["Não fraudulenta", "Fraudulenta"]
ax.set_xticks([0, 1]); ax.set_xticklabels(rot); ax.set_yticks([0, 1]); ax.set_yticklabels(rot)
ax.set_xlabel("Previsto"); ax.set_ylabel("Real"); ax.set_title("Matriz de confusão (teste)")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12,
                color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")

ax = axes[1]
for nome, p, cor in [("Gradient Boosting", p_te_gb, AZUL), ("Regressão Logística", p_te_log, LARANJA)]:
    pr, rc, _ = precision_recall_curve(y[teste], p)
    ax.plot(rc, pr, color=cor, lw=2, label=f"{nome} (PR-AUC {average_precision_score(y[teste], p):.3f})")
ax.axhline(y[teste].mean(), color=CINZA, ls="--", lw=1, label="Aleatório")
ax.set_xlabel("Recall (fraudes detectadas)"); ax.set_ylabel("Precisão (alertas corretos)")
ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
ax.set_title("Curva Precisão-Recall (teste)"); ax.legend(frameon=False, loc="upper right")
plt.tight_layout(); plt.show()
# Visão de negócio: se a equipe revisar apenas as X% transações de maior risco, quantas fraudes encontra?
ordem = np.argsort(-p_te_gb)
y_ord = y[teste].values[ordem]
linhas = []
for frac in [0.01, 0.02, 0.05, 0.10, 0.20]:
    k = int(len(y_ord) * frac)
    linhas.append({"transações revisadas": f"top {frac:.0%}", "qtd revisada": k,
                   "fraudes encontradas": int(y_ord[:k].sum()),
                   "% das fraudes capturadas": y_ord[:k].sum() / y_ord.sum(),
                   "% de fraude entre as revisadas": y_ord[:k].mean()})
pd.DataFrame(linhas).set_index("transações revisadas").style.format(
    {"qtd revisada": "{:,}", "fraudes encontradas": "{:,}",
     "% das fraudes capturadas": "{:.1%}", "% de fraude entre as revisadas": "{:.1%}"})
     # Importância das variáveis (permutação no teste, métrica PR-AUC)
amostra = X[teste].sample(40_000, random_state=SEED)
imp = permutation_importance(gb, amostra, y.loc[amostra.index], scoring="average_precision",
                             n_repeats=5, random_state=SEED, n_jobs=-1)
imp = pd.Series(imp.importances_mean, index=amostra.columns).sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(imp.index, imp.values, color=[AZUL if v > 0.005 else "#c9c8c2" for v in imp.values], height=0.6)
ax.set_xlabel("Queda na PR-AUC ao embaralhar a variável"); ax.grid(axis="y", visible=False)
ax.set_title("Importância das variáveis (Gradient Boosting)")
plt.show()
modelo_final = HistGradientBoostingClassifier(**PARAMS_GB).fit(X, y)

X_jan = matriz_features(jan)
jan["prob_fraude"] = modelo_final.predict_proba(X_jan)[:, 1]
jan["classificacao"] = np.where(jan["prob_fraude"] >= LIMIAR, "fraudulenta", "não fraudulenta")

n_fr = (jan["classificacao"] == "fraudulenta").sum()
print(f"Transações em jan/2026         : {len(jan):,}")
print(f"Classificadas como fraudulentas: {n_fr:,} ({n_fr/len(jan):.2%})")
print(f"Fraudes esperadas (soma das probabilidades): {jan['prob_fraude'].sum():,.0f} ({jan['prob_fraude'].mean():.2%})")
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
ax = axes[0]
ax.hist(jan["prob_fraude"], bins=50, color=AZUL)
ax.axvline(LIMIAR, color=LARANJA, lw=2); ax.text(LIMIAR, ax.get_ylim()[1]*0.9, f" limiar {LIMIAR:.2f}", color="#52514e")
ax.set_yscale("log"); ax.set_xlabel("Probabilidade de fraude"); ax.set_ylabel("Transações (escala log)")
ax.set_title("Distribuição do risco em jan/2026")

ax = axes[1]
fh = jan.groupby("hora")["classificacao"].apply(lambda s: (s == "fraudulenta").sum())
ax.bar(fh.index, fh.values, color=AZUL, width=0.8)
ax.set_xticks(range(24)); ax.set_xlabel("Hora"); ax.set_ylabel("Transações")
ax.set_title("Transações classificadas como fraudulentas por hora (jan/2026)")
plt.tight_layout(); plt.show()

(jan[jan.classificacao == "fraudulenta"]
   .groupby("tipo_estabelecimento").size().sort_values(ascending=False)
   .rename("fraudes previstas").to_frame()
   .assign(**{"% do total previsto": lambda d: d["fraudes previstas"] / d["fraudes previstas"].sum()})
   .style.format({"% do total previsto": "{:.1%}"}))
   # 20 transações de maior risco em janeiro/2026
cols_show = ["data", "HORARIO", "nome_estabelecimento", "tipo_estabelecimento", "valor_transacao",
             "cliente_id", "cidade", "device_type", "is_new_device", "prob_fraude", "classificacao"]
jan.sort_values("prob_fraude", ascending=False)[cols_show].head(20).reset_index(drop=True)
# Exporta o resultado
saida = jan.drop(columns=["minuto", "datahora", "dia_semana", "dia_mes", "mes"]).sort_values(["data", "HORARIO"])
saida["prob_fraude"] = saida["prob_fraude"].round(4)
saida.to_excel("transacoes_janeiro_2026_classificadas.xlsx", index=False)
print("Arquivo gerado: transacoes_janeiro_2026_classificadas.xlsx", saida.shape)

ARQ_JAN  = "transacoes-janeiro-nivea.xlsx"  # janeiro/2026 sem rótulo
SEED = 42
