import sys
import os
import tempfile

sys.path.insert(0, ".")

from app.indexing.inverted_index import InvertedIndex


def test_empty_index():
    idx = InvertedIndex()

    assert idx.doc_count == 0
    assert idx.get_vocabulary_size() == 0
    assert idx.get_postings("anything") == []
    assert idx.get_document_frequency("anything") == 0
    print("✅ empty_index: OK")


def test_add_document():
    idx = InvertedIndex()

    term_positions = {"moon": [0, 5], "astronaut": [1], "land": [2, 3]}
    idx.add_document("doc1", term_positions)

    assert idx.doc_count == 1
    assert idx.get_vocabulary_size() == 3
    assert idx.contains_term("moon")
    assert idx.contains_term("astronaut")
    assert idx.contains_term("land")
    assert idx.get_postings("moon")[0].doc_id == "doc1"
    assert idx.get_postings("moon")[0].positions == [0, 5]
    assert idx.get_postings("astronaut")[0].doc_id == "doc1"
    assert idx.get_postings("astronaut")[0].positions == [1]
    assert idx.get_postings("land")[0].doc_id == "doc1"
    assert idx.get_postings("land")[0].positions == [2, 3]
    print("✅ add_document: OK")


def test_posting_data():
    idx = InvertedIndex()

    term_positions = {"moon": [0, 5, 10], "nasa": [2]}
    idx.add_document("doc1", term_positions)

    moon_postings = idx.get_postings("moon")
    assert len(moon_postings) == 1
    assert moon_postings[0].doc_id == "doc1"
    assert moon_postings[0].tf == 3
    assert moon_postings[0].positions == [0, 5, 10]

    nasa_postings = idx.get_postings("nasa")
    assert nasa_postings[0].tf == 1
    print("✅ posting_data: OK")


def test_document_frequency():
    idx = InvertedIndex()

    idx.add_document("doc1", {"moon": [0], "nasa": [1]})
    idx.add_document("doc2", {"moon": [0], "apollo": [1]})
    idx.add_document("doc3", {"apollo": [0]})

    assert idx.get_document_frequency("moon") == 2
    assert idx.get_document_frequency("nasa") == 1
    assert idx.get_document_frequency("apollo") == 2
    assert idx.get_document_frequency("unknown") == 0
    print("✅ document_frequency: OK")


def test_term_frequency():
    idx = InvertedIndex()

    idx.add_document("doc1", {"moon": [0, 5, 10]})
    idx.add_document("doc2", {"moon": [0]})

    assert idx.get_term_frequency("moon", "doc1") == 3
    assert idx.get_term_frequency("moon", "doc2") == 1
    assert idx.get_term_frequency("moon", "doc3") == 0
    assert idx.get_term_frequency("unknown", "doc1") == 0
    print("✅ term_frequency: OK")


def test_vocabulary():
    idx = InvertedIndex()

    idx.add_document("doc1", {"moon": [0], "nasa": [1], "apollo": [2]})

    vocab = idx.get_vocabulary()
    assert len(vocab) == 3
    assert "moon" in vocab
    assert "nasa" in vocab
    assert "apollo" in vocab
    print("✅ vocabulary: OK")


def test_docs_containing_term():
    idx = InvertedIndex()

    idx.add_document("doc1", {"moon": [0]})
    idx.add_document("doc2", {"moon": [0], "nasa": [1]})
    idx.add_document("doc3", {"nasa": [0]})

    moon_docs = idx.get_docs_containing_term("moon")
    assert "doc1" in moon_docs
    assert "doc2" in moon_docs
    assert "doc3" not in moon_docs

    nasa_docs = idx.get_docs_containing_term("nasa")
    assert len(nasa_docs) == 2
    print("✅ docs_containing_term: OK")


def test_doc_lengths():
    idx = InvertedIndex()

    idx.add_document("doc1", {"moon": [0, 1, 2], "nasa": [3, 4]})  # 5 términos
    idx.add_document("doc2", {"apollo": [0]})  # 1 término

    assert idx.doc_lengths["doc1"] == 5
    assert idx.doc_lengths["doc2"] == 1
    assert idx.get_average_doc_length() == 3.0
    print("✅ doc_lengths: OK")


def test_save_and_load():
    idx = InvertedIndex()

    idx.add_document("doc1", {"moon": [0, 5], "nasa": [1]})
    idx.add_document("doc2", {"moon": [0], "apollo": [2, 3]})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        idx.save(temp_path)

        idx2 = InvertedIndex()
        idx2.load(temp_path)

        assert idx2.doc_count == 2
        assert idx2.get_vocabulary_size() == 3
        assert idx2.get_document_frequency("moon") == 2
        assert idx2.get_term_frequency("moon", "doc1") == 2
        assert idx2.doc_lengths["doc1"] == 3

        moon_postings = idx2.get_postings("moon")
        assert moon_postings[0].positions == [0, 5]

        print("✅ save_and_load: OK")
    finally:
        os.unlink(temp_path)


def test_clear():
    idx = InvertedIndex()

    idx.add_document("doc1", {"moon": [0]})
    assert idx.doc_count == 1

    idx.clear()

    assert idx.doc_count == 0
    assert idx.get_vocabulary_size() == 0
    assert idx.doc_lengths == {}
    print("✅ clear: OK")


if __name__ == "__main__":
    print("\n🧪 Testing InvertedIndex\n" + "=" * 40)

    test_empty_index()
    test_add_document()
    test_posting_data()
    test_document_frequency()
    test_term_frequency()
    test_vocabulary()
    test_docs_containing_term()
    test_doc_lengths()
    test_save_and_load()
    test_clear()

    print("\n" + "=" * 40)
    print("✅ Todos los tests pasaron!")
