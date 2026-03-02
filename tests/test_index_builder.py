import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, ".")

from app.indexing.index_builder import IndexBuilder


def create_test_data(temp_dir):
    doc1 = {
        "id": "doc1",
        "title": "NASA Moon Mission",
        "content": "Astronauts landed on the Moon in 1969. The Moon landing was historic.",
        "source": "BBC",
        "category": "Science",
        "url": "http://example.com/1",
        "author": "John Doe",
        "date": "2024-01-01",
    }

    doc2 = {
        "id": "doc2",
        "title": "Mars Exploration",
        "content": "NASA plans to send astronauts to Mars. The mission will be challenging.",
        "source": "CNN",
        "category": "Technology",
        "url": "http://example.com/2",
        "author": "Jane Doe",
        "date": "2024-01-02",
    }

    for doc in [doc1, doc2]:
        filepath = os.path.join(temp_dir, f"{doc['id']}.json")
        with open(filepath, "w") as f:
            json.dump(doc, f)

    return [doc1, doc2]


def test_load_document():
    with tempfile.TemporaryDirectory() as temp_dir:
        doc = {"id": "test", "title": "Test", "content": "Content"}
        filepath = os.path.join(temp_dir, "test.json")
        with open(filepath, "w") as f:
            json.dump(doc, f)

        builder = IndexBuilder(data_path=temp_dir)
        loaded = builder.load_document(filepath)

        assert loaded["id"] == "test"
        assert loaded["title"] == "Test"
        print("✅ load_document: OK")


def test_get_document_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        create_test_data(temp_dir)

        builder = IndexBuilder(data_path=temp_dir)
        files = builder.get_document_files()

        assert len(files) == 2
        assert all(f.endswith(".json") for f in files)
        print("✅ get_document_files: OK")


def test_extract_indexable_text():
    builder = IndexBuilder()

    doc = {"title": "Moon Landing", "content": "Historic event"}
    text = builder.extract_indexable_text(doc)

    assert "Moon Landing" in text
    assert "Historic event" in text
    print("✅ extract_indexable_text: OK")


def test_extract_metadata():
    builder = IndexBuilder()

    doc = {
        "id": "123",
        "title": "Test Title",
        "source": "BBC",
        "category": "Science",
        "url": "http://example.com",
        "author": "Author",
        "date": "2024-01-01",
        "content": "Content here",
    }

    metadata = builder.extract_metadata(doc)

    assert metadata["title"] == "Test Title"
    assert metadata["source"] == "BBC"
    assert "content" not in metadata
    assert "id" not in metadata
    print("✅ extract_metadata: OK")


def test_index_document():
    with tempfile.TemporaryDirectory() as temp_dir:
        builder = IndexBuilder(data_path=temp_dir)

        doc = {
            "id": "doc1",
            "title": "Moon",
            "content": "The Moon is beautiful. Moon landing.",
        }

        builder.index_document(doc)

        assert builder.index.doc_count == 1
        assert builder.index.contains_term("moon")
        assert builder.index.get_document_frequency("moon") == 1
        assert "doc1" in builder.documents_metadata
        print("✅ index_document: OK")


def test_build():
    with tempfile.TemporaryDirectory() as temp_dir:
        create_test_data(temp_dir)

        builder = IndexBuilder(data_path=temp_dir)
        count = builder.build()

        assert count == 2
        assert builder.index.doc_count == 2
        assert builder.index.contains_term("moon")
        assert builder.index.contains_term("mar")
        assert builder.index.contains_term("nasa")
        assert builder.index.contains_term("astronaut")
        print("✅ build: OK")


def test_save_and_load():
    with tempfile.TemporaryDirectory() as data_dir:
        with tempfile.TemporaryDirectory() as index_dir:
            create_test_data(data_dir)

            builder1 = IndexBuilder(data_path=data_dir, index_path=index_dir)
            builder1.build()
            builder1.save()

            builder2 = IndexBuilder(data_path=data_dir, index_path=index_dir)
            builder2.load()

            assert builder2.index.doc_count == 2
            assert (
                builder2.index.get_vocabulary_size()
                == builder1.index.get_vocabulary_size()
            )
            assert len(builder2.documents_metadata) == 2
            assert builder2.documents_metadata["doc1"]["title"] == "NASA Moon Mission"
            print("✅ save_and_load: OK")


def test_get_document_metadata():
    with tempfile.TemporaryDirectory() as temp_dir:
        create_test_data(temp_dir)

        builder = IndexBuilder(data_path=temp_dir)
        builder.build()

        metadata = builder.get_document_metadata("doc1")

        assert metadata is not None
        assert metadata["title"] == "NASA Moon Mission"
        assert metadata["source"] == "BBC"

        assert builder.get_document_metadata("nonexistent") is None
        print("✅ get_document_metadata: OK")


def test_get_stats():
    with tempfile.TemporaryDirectory() as temp_dir:
        create_test_data(temp_dir)

        builder = IndexBuilder(data_path=temp_dir)
        builder.build()

        stats = builder.get_stats()

        assert stats["total_documents"] == 2
        assert stats["vocabulary_size"] > 0
        assert stats["avg_doc_length"] > 0
        print("✅ get_stats: OK")


if __name__ == "__main__":
    print("\n🧪 Testing IndexBuilder\n" + "=" * 40)

    test_load_document()
    test_get_document_files()
    test_extract_indexable_text()
    test_extract_metadata()
    test_index_document()
    test_build()
    test_save_and_load()
    test_get_document_metadata()
    test_get_stats()

    print("\n" + "=" * 40)
    print("✅ Todos los tests pasaron!")
