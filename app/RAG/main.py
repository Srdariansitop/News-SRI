import os
import json
from app.RAG.rag import RAGSystem

# Mock (Simulador) del buscador para pruebas rápidas
class MockHybridSearcher:
    def search(self, query: str, top_k: int = 10) -> list:
        print(f"   [Mock] Buscando '{query}' en el índice...")
        return [
            {"doc_id": "test_1", "score": 0.9, "title": "Noticia Espacial", "url": "http://fake-news.com/1"},
            {"doc_id": "test_2", "score": 0.8, "title": "Agricultura Lunar", "url": "http://fake-news.com/2"},
            {"doc_id": "test_3", "score": 0.7, "title": "Sin contenido", "url": "http://fake-news.com/3"}
        ]

def main():
    print("="*50)
    print("🚀 TEST DEL SISTEMA RAG CON OPENROUTER")
    print("="*50)
    
    # 1. Crear la carpeta data/raw por si acaso
    os.makedirs("data/raw", exist_ok=True)
    
    # 2. Generar documentos de prueba en data/raw (JSON falsos para que el RAG los lea)
    doc1_path = "data/raw/test_1.json"
    with open(doc1_path, "w", encoding="utf-8") as f:
        json.dump({"content": "La capital de la Luna es la Base Lunar Alpha. La NASA confirmó que la estructura se mantiene con paneles solares gigantes instalados en los cráteres."}, f)
        
    doc2_path = "data/raw/test_2.json"
    with open(doc2_path, "w", encoding="utf-8") as f:
        json.dump({"content": "La principal exportación de la Luna a la Tierra en el año 2045 es el isótopo Helio-3, el cual ha revolucionado los reactores de fusión nuclear terrestres."}, f)
        
    # (El test_3 no lo creamos para comprobar que el código maneje documentos sin contenido)

    # 3. Inicialización del RAG
    print("[*] Conectando con IA distribuida...")
    mock_searcher = MockHybridSearcher()
    rag = RAGSystem(hybrid_searcher=mock_searcher, raw_data_path="data/raw")

    # 4. Hacer la prueba de pregunta
    pregunta = "¿Cuál es la capital de la Luna y qué exporta a la Tierra?"
    print(f"\n👤 Usuario: {pregunta}")
    
    print("[*] Generando respuesta con RAG (esto puede tardar unos segundos)...")
    try:
        resultado = rag.answer(pregunta)
        
        print("\n" + "="*50)
        print("🤖 RESPUESTA DE LA IA:")
        print(resultado["summary"])
        print("="*50)
        
        print("\n📚 Documentos consultados:")
        for doc in resultado["documents"]:
            print(f" - [{doc['doc_id']}] {doc['title']} ({doc['url']})")
    
    except Exception as e:
        print(f"\n❌ Algo falló en la prueba: {e}")

if __name__ == "__main__":
    main()
