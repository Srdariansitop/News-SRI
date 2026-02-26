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
MAX_ARTICLES_PER_CATEGORY = 20

# Crear una sesión global para guardar cookies y parecer humano
session = requests.Session()
session.headers.update(HEADERS)


def ensure_data_folder():
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)

def generate_document_id():
    return str(uuid.uuid4())

def clean_text(text):
    return " ".join(text.split())



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
                document = {
                    "id": generate_document_id(),
                    "source": "BBC",
                    "category": category,
                    "url": url,
                    "title": article_data["title"],
                    "author": article_data["author"],
                    "date": article_data["date"],
                    "content": article_data["content"],
                    # CORRECCIÓN DE FECHA
                    "crawled_at": datetime.now(UTC).isoformat() 
                }

                filename = os.path.join(RAW_DATA_PATH, f"{document['id']}.json")

                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(document, f, ensure_ascii=False, indent=4)

                print(f"✅ Guardado: {filename}")

            # DELAY ALEATORIO: Entre 2 y 4 segundos para evitar bloqueos
            time.sleep(random.uniform(2, 4))


if __name__ == "__main__":
    crawl()