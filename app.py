from flask import Flask, render_template, request
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode

app = Flask(__name__)

# 🔹 Limpar texto
def limpar_texto(texto):
    return unidecode(texto.lower())

# 🔹 Buscar licitações
def buscar_licitacoes(termo, uf=None, municipio=None, valor_min=0):
    url = "https://pncp.gov.br/api/search/"
    
    descricoes = []
    registros = []

    for pagina in range(1, 6):
        params = {
            "q": termo,
            "tipos_documento": "edital",
            "pagina": pagina,
            "tam_pagina": 100
        }

        if uf:
            params["ufs"] = uf.upper()

        response = requests.get(url, params=params)

        if response.status_code != 200:
            continue

        data = response.json()
        itens = data.get("items", [])

        if not itens:
            break

        for item in itens:
            descricao = item.get("description") or item.get("title") or ""

            if not descricao or len(descricao) < 15:
                continue

            # filtro município
            if municipio:
                if municipio.lower() not in (item.get("municipio_nome") or "").lower():
                    continue

            valor = item.get("valor_global") or 0

            if valor < valor_min:
                continue

            descricoes.append(limpar_texto(descricao))
            registros.append(item)

    # remover duplicados
    vistos = set()
    desc_final = []
    reg_final = []

    for d, r in zip(descricoes, registros):
        if d not in vistos:
            vistos.add(d)
            desc_final.append(d)
            reg_final.append(r)

    return desc_final, reg_final

# 🔹 Buscar similares
def buscar_similares(consulta, descricoes, registros):
    if len(descricoes) < 2:
        return [], None

    consulta = limpar_texto(consulta)

    vectorizer = TfidfVectorizer()
    matriz = vectorizer.fit_transform(descricoes + [consulta])

    similaridade = cosine_similarity(matriz[-1], matriz[:-1])

    resultados = list(enumerate(similaridade[0]))
    resultados.sort(key=lambda x: x[1], reverse=True)

    top = []
    valores = []

    for i, score in resultados[:5]:
        item = registros[i]

        valor = item.get("valor_global")

        if isinstance(valor, (int, float)):
            valores.append(valor)

        url_original = item.get("item_url") or ""

        # corrige o caminho
        url_corrigida = url_original.replace("/compras/", "/app/editais/")

        link = "https://pncp.gov.br" + url_corrigida

        top.append({
            "similaridade": round(score, 2),
            "descricao": item.get("description") or item.get("title"),
            "orgao": item.get("orgao_nome", "N/A"),
            "municipio": item.get("municipio_nome", "N/A"),
            "uf": item.get("uf", ""),
            "valor": valor,
            "link": link
        })

    stats = None
    if valores:
        stats = {
            "media": f"{sum(valores)/len(valores):,.2f}",
            "menor": f"{min(valores):,.2f}",
            "maior": f"{max(valores):,.2f}"
        }

    return top, stats

# 🔹 Rota principal
@app.route("/")
def index():
    consulta = request.args.get("consulta")
    uf = request.args.get("uf")
    municipio = request.args.get("municipio")
    valor_min = request.args.get("valor_min")

    valor_min = float(valor_min) if valor_min else 0

    resultados = []
    stats = None

    if consulta:
        descricoes, registros = buscar_licitacoes(consulta, uf, municipio, valor_min)
        resultados, stats = buscar_similares(consulta, descricoes, registros)

    return render_template("index.html", resultados=resultados, stats=stats)

if __name__ == "__main__":
    app.run(debug=True)