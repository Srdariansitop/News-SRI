import os
import json
import time
import uuid
import random
import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, UTC


# Nuevos Feeds de la BBC
RSS_FEEDS = {
    "Technology": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Business": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "Science": "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "Politics": "http://feeds.bbci.co.uk/news/politics/rss.xml"
}

# Headers simulando ser un navegador Chrome real
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
}

RAW_DATA_PATH = "data/raw"
MAX_ARTICLES_PER_CATEGORY = 5

# Crear una sesión global para guardar cookies y parecer humano
session = requests.Session()
session.headers.update(HEADERS)


def ensure_data_folder():
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)

def ensure_ranking_metrics():
    """
    Asegura que popularity_metrics.json exista.
    Se crea vacío si fue borrado o no existe.
    """
    ranking_path = "data/ranking"
    metrics_file = os.path.join(ranking_path, "popularity_metrics.json")
    
    if not os.path.exists(ranking_path):
        os.makedirs(ranking_path)
    
    if not os.path.exists(metrics_file):
        default_metrics = {
            "_metadata": {
                "description": "Almacena métricas de popularidad de documentos para el módulo de posicionamiento",
                "created_at": datetime.now(UTC).isoformat(),
                "structure": {
                    "doc_id": {
                        "clicks": "Cuántas veces fue mostrado al usuario (click en resultado)",
                        "search_frequency": "En cuántas búsquedas únicas apareció este documento",
                        "shares": "Cuántas veces fue compartido",
                        "last_updated": "Última vez que se actualizó la métrica"
                    }
                }
            },
            "_example": {
                "00104b6f-0fbb-4306-b46f-dcd20317970e": {
                    "clicks": 0,
                    "search_frequency": 0,
                    "shares": 0,
                    "last_updated": datetime.now(UTC).isoformat()
                }
            }
        }
        
        try:
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(default_metrics, f, indent=2, ensure_ascii=False)
            print(f"📊 Archivo de métricas creado: {metrics_file}")
        except Exception as e:
            print(f"⚠️ Error creando popularity_metrics.json: {e}")

def generate_document_id():
    return str(uuid.uuid4())

def clean_text(text):
    return " ".join(text.split())



def detect_content_type(title: str, content: str) -> str:
    """
    Detecta el tipo de contenido según palabras clave en título y contenido.
    Returns: "breaking_news", "news", "analysis", "opinion"
    """
    title_lower = title.lower()
    content_lower = content.lower()
    
    # BREAKING NEWS
    if any(word in title_lower for word in ["breaking", "just in", "live", "alert"]):
        return "breaking_news"
    
    # ANALYSIS / OPINION (palabras clave)
    if any(word in title_lower for word in ["analysis", "opinion", "why", "how to", "explained", "what is", "guide"]):
        return "analysis"
    
    # Por defecto: NOTICIA
    return "news"


def scrape_article(url):
    try:
        response = session.get(url, timeout=10)

        if response.status_code != 200:
            print(f"❌ Error HTTP {response.status_code} en {url}")
            return None

        soup = BeautifulSoup(response.text, "lxml")

        # Título: La BBC suele usar un h1 principal
        title_tag = soup.find("h1")
        title = clean_text(title_tag.get_text()) if title_tag else ""

        # Autor: La BBC a veces usa clases que contienen 'byline' o no pone autor.
        author_tag = soup.find(attrs={"data-testid": "byline-name"})
        author = clean_text(author_tag.get_text()) if author_tag else "BBC News"

        # Fecha: Usan la etiqueta time
        time_tag = soup.find("time")
        date = time_tag.get("datetime") if time_tag else ""

        # Contenido: La BBC envuelve las noticias en un <article>
        article_body = soup.find("article")
        if article_body:
            paragraphs = article_body.find_all("p")
        else:
            # Si no hay etiqueta article, buscamos todos los párrafos de la página
            paragraphs = soup.find_all("p")

        content = " ".join([clean_text(p.get_text()) for p in paragraphs])

        # Ignoramos si extrajo muy poco texto (probablemente una galería o video)
        if len(content) < 300:
            print(f"⚠️ Contenido muy corto o es un video en {url}")
            return None

        return {
            "title": title,
            "author": author,
            "date": date,
            "content": content
        }

    except Exception as e:
        print(f"❌ Excepción en {url}: {e}")
        return None



def crawl():
    ensure_data_folder()
    ensure_ranking_metrics()

    for category, feed_url in RSS_FEEDS.items():
        print(f"\n📡 Procesando categoría: {category}")

        feed = feedparser.parse(feed_url)
        entries = feed.entries[:MAX_ARTICLES_PER_CATEGORY]

        for entry in entries:
            url = entry.link
            
            # Evitar enlaces a videos en vivo de la BBC que no tienen formato de artículo
            if "/live/" in url or "/av/" in url:
                continue
                
            print(f"🔎 Descargando: {url}")

            article_data = scrape_article(url)

            if article_data:
                # Detectar tipo de contenido para posicionamiento
                content_type = detect_content_type(article_data["title"], article_data["content"])
                is_breaking = content_type == "breaking_news"
                
                document = {
                    "id": generate_document_id(),
                    "source": "BBC",
                    "category": category,
                    "url": url,
                    "title": article_data["title"],
                    "author": article_data["author"],
                    "date": article_data["date"],
                    "content": article_data["content"],
                    "crawled_at": datetime.now(UTC).isoformat(),
                    
                    # NUEVOS CAMPOS PARA POSICIONAMIENTO
                    "domain_authority": 0.95,      # BBC es altamente confiable
                    "content_type": content_type,  # breaking_news, news, analysis, opinion
                    "is_breaking": is_breaking     # True si es noticia de última hora
                }

                filename = os.path.join(RAW_DATA_PATH, f"{document['id']}.json")

                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(document, f, ensure_ascii=False, indent=4)

                print(f"✅ Guardado: {filename} [Tipo: {content_type}]")

            # DELAY ALEATORIO: Entre 2 y 4 segundos para evitar bloqueos
            time.sleep(random.uniform(2, 4))


if __name__ == "__main__":
    crawl()