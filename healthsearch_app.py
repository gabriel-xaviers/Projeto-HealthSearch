import re
import math
import streamlit as st
import pandas as pd

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="HealthSearch",
    page_icon="🔎",
    layout="wide"
)

# ============================================================
# CORPUS OBRIGATÓRIO DO DESAFIO
# ============================================================

DOCUMENTS = [
    {
        "id": "Doc 1",
        "title": "Protocolo Emergência ECG",
        "content": (
            "Pacientes com dor precordial aguda e suspeita de síndrome "
            "coronariana devem realizar eletrocardiograma CÓD-ECG-12D "
            "em até 10 minutos."
        ),
    },
    {
        "id": "Doc 2",
        "title": "Guia de Farmacologia Cardíaca",
        "content": (
            "O uso imediato de ácido acetilsalicílico e antiagregantes "
            "plaquetários reduz a mortalidade no infarto agudo do miocárdio."
        ),
    },
    {
        "id": "Doc 3",
        "title": "Diretriz de Hipertensão Arterial",
        "content": (
            "A crise hipertensiva severa requer administração de "
            "anti-hipertensivos venosos e monitoramento contínuo da "
            "pressão arterial na UTI."
        ),
    },
    {
        "id": "Doc 4",
        "title": "Manual de AVC Isquêmico",
        "content": (
            "O acidente vascular cerebral isquêmico agudo deve ser tratado "
            "com trombolíticos venosos em até quatro horas e meia do início "
            "dos sintomas."
        ),
    },
    {
        "id": "Doc 5",
        "title": "Protocolo de Reanimação RCR",
        "content": (
            "Parada cardiorrespiratória em adultos exige compressões "
            "torácicas contínuas de alta qualidade e desfibrilação precoce "
            "no código azul."
        ),
    },
    {
        "id": "Doc 6",
        "title": "Procedimentos de UTI Geral",
        "content": (
            "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias "
            "complexas, recomenda-se a monitorização cardíaca contínua "
            "por telemetria."
        ),
    },
]

STOPWORDS = {
    "a", "à", "ao", "aos", "as", "às", "com", "como", "da", "das", "de",
    "do", "dos", "e", "em", "entre", "era", "esse", "esta", "este", "foi",
    "há", "na", "nas", "no", "nos", "o", "os", "ou", "para", "por", "que",
    "se", "sem", "sua", "suas", "um", "uma", "uns", "umas", "é", "ser",
    "até", "mais", "sobre", "também", "pelo", "pela", "pelos", "pelas",
    "após", "dos"
}

# Mantém códigos médicos como um único token.
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9áàâãéêíóôõúçü\-]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]

TOKENS = [tokenize(d["content"]) for d in DOCUMENTS]

# ============================================================
# MODELO SEMÂNTICO
# ============================================================

@st.cache_resource
def load_model():
    # Modelo multilíngue, adequado para consultas em português.
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def build_embeddings():
    model = load_model()
    texts = [d["title"] + ". " + d["content"] for d in DOCUMENTS]
    return model.encode(texts, normalize_embeddings=True)

# ============================================================
# FUNÇÕES DE BUSCA
# ============================================================

def bm25_search(query, k1, b):
    bm25 = BM25Okapi(TOKENS, k1=k1, b=b)
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)

    rows = []
    for i, score in enumerate(scores):
        rows.append({
            "id": DOCUMENTS[i]["id"],
            "title": DOCUMENTS[i]["title"],
            "content": DOCUMENTS[i]["content"],
            "score": float(score),
            "index": i
        })

    rows.sort(key=lambda x: x["score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows

def semantic_search(query):
    model = load_model()
    embeddings = build_embeddings()
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(query_embedding, embeddings)[0]

    rows = []
    for i, score in enumerate(scores):
        rows.append({
            "id": DOCUMENTS[i]["id"],
            "title": DOCUMENTS[i]["title"],
            "content": DOCUMENTS[i]["content"],
            "score": float(score),
            "index": i
        })

    rows.sort(key=lambda x: x["score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows

def rrf_search(bm25_rows, semantic_rows, alpha, k_rrf=60):
    bm25_rank = {r["index"]: r["rank"] for r in bm25_rows}
    sem_rank = {r["index"]: r["rank"] for r in semantic_rows}

    rows = []
    for i, doc in enumerate(DOCUMENTS):
        score = (
            alpha * (1 / (k_rrf + bm25_rank[i]))
            + (1 - alpha) * (1 / (k_rrf + sem_rank[i]))
        )
        rows.append({
            "id": doc["id"],
            "title": doc["title"],
            "content": doc["content"],
            "score": float(score),
            "index": i,
            "rank_bm25": bm25_rank[i],
            "rank_semantic": sem_rank[i],
        })

    rows.sort(key=lambda x: x["score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank_rrf"] = rank
    return rows

def rows_to_df(rows, score_name="Score"):
    data = []
    for r in rows:
        data.append({
            "Rank": r.get("rank", r.get("rank_rrf")),
            "ID": r["id"],
            "Título": r["title"],
            score_name: round(r["score"], 6),
        })
    return pd.DataFrame(data)

# ============================================================
# INTERFACE
# ============================================================

st.title("🔎 HealthSearch")
st.caption("Motor de Busca Híbrido — BM25 + Busca Semântica + RRF")

st.sidebar.header("⚙️ Parâmetros")

k1 = st.sidebar.slider(
    "k₁ — Saturação de frequência",
    min_value=0.0,
    max_value=3.0,
    value=1.2,
    step=0.1
)

b = st.sidebar.slider(
    "b — Normalização pelo comprimento",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05
)

alpha = st.sidebar.slider(
    "α — Peso do BM25 no RRF",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.info(
    "O RRF utiliza k_RRF = 60, conforme especificado no desafio."
)

query = st.text_input(
    "🔍 Consulta médica",
    value="ataque cardíaco",
    placeholder="Ex.: ataque cardíaco, ECG-12D, pressão alta..."
)

if not query.strip():
    st.warning("Digite uma consulta para executar a busca.")
    st.stop()

with st.spinner("Executando BM25 e busca semântica..."):
    bm25_rows = bm25_search(query, k1, b)
    semantic_rows = semantic_search(query)
    hybrid_rows = rrf_search(bm25_rows, semantic_rows, alpha)

tab1, tab2, tab3, tab4 = st.tabs([
    "📖 Léxico — BM25",
    "🧠 Semântico",
    "🔀 Híbrido — RRF",
    "📊 Matriz Comparativa"
])

# ------------------------------------------------------------
# ABA BM25
# ------------------------------------------------------------

with tab1:
    st.subheader("Busca Léxica — Okapi BM25")
    st.write(
        "O BM25 privilegia correspondências lexicais entre os termos da "
        "consulta e os documentos."
    )

    st.metric("k₁", f"{k1:.1f}")
    st.metric("b", f"{b:.2f}")

    df = rows_to_df(bm25_rows, "Score BM25")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Detalhamento")
    for r in bm25_rows:
        with st.expander(f"{r['rank']}º — {r['id']} | {r['title']}"):
            st.write(r["content"])
            st.write(f"**Score BM25:** {r['score']:.6f}")

# ------------------------------------------------------------
# ABA SEMÂNTICA
# ------------------------------------------------------------

with tab2:
    st.subheader("Busca Semântica Vetorial")
    st.write(
        "Os documentos e a consulta são representados por embeddings e "
        "comparados usando similaridade de cosseno."
    )

    df = rows_to_df(semantic_rows, "Similaridade de Cosseno")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Detalhamento")
    for r in semantic_rows:
        with st.expander(f"{r['rank']}º — {r['id']} | {r['title']}"):
            st.write(r["content"])
            st.write(f"**Similaridade:** {r['score']:.6f}")

# ------------------------------------------------------------
# ABA RRF
# ------------------------------------------------------------

with tab3:
    st.subheader("Busca Híbrida — Reciprocal Rank Fusion")

    st.latex(
        r"Score_{RRF}(D)=\alpha\frac{1}{k_{RRF}+Rank_{BM25}}"
        r"+(1-\alpha)\frac{1}{k_{RRF}+Rank_{Semântico}}"
    )

    st.write(
        f"α = **{alpha:.2f}** | k_RRF = **60**"
    )

    hybrid_df = pd.DataFrame([
        {
            "Rank RRF": r["rank_rrf"],
            "ID": r["id"],
            "Título": r["title"],
            "Rank BM25": r["rank_bm25"],
            "Rank Semântico": r["rank_semantic"],
            "Score RRF": round(r["score"], 8)
        }
        for r in hybrid_rows
    ])

    st.dataframe(
        hybrid_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Resultados híbridos")
    for r in hybrid_rows:
        with st.expander(
            f"{r['rank_rrf']}º — {r['id']} | {r['title']}"
        ):
            st.write(r["content"])
            st.write(
                f"**RRF:** {r['score']:.8f} | "
                f"BM25 rank: {r['rank_bm25']} | "
                f"Semântico rank: {r['rank_semantic']}"
            )

# ------------------------------------------------------------
# ABA COMPARATIVA
# ------------------------------------------------------------

with tab4:
    st.subheader("Matriz Comparativa de Rankings")

    comparison = pd.DataFrame([
        {
            "Documento": r["id"],
            "Título": r["title"],
            "Rank BM25": bm25_rows[r["index"]]["rank"],
            "Rank Semântico": semantic_rows[r["index"]]["rank"],
            "Rank RRF": r["rank_rrf"],
            "Score BM25": round(bm25_rows[r["index"]]["score"], 5),
            "Score Semântico": round(semantic_rows[r["index"]]["score"], 5),
            "Score RRF": round(r["score"], 7),
        }
        for r in hybrid_rows
    ]).sort_values("Rank RRF")

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Gráfico de comparação de ranks")
    chart_df = comparison.set_index("Documento")[
        ["Rank BM25", "Rank Semântico", "Rank RRF"]
    ]
    st.bar_chart(chart_df)

st.markdown("---")
st.caption(
    "HealthSearch — protótipo acadêmico para demonstração de recuperação "
    "de informação híbrida. Não substitui protocolos médicos reais."
)
