import sys

sys.path.insert(0, ".")

from app.core.preprocessor import Preprocessor, Token, get_preprocessor


def test_normalize():
    prep = Preprocessor()

    assert prep.normalize("Hello World") == "hello world"
    assert prep.normalize("NASA Apollo") == "nasa apollo"
    assert prep.normalize("") == ""
    print("✅ normalize: OK")


def test_tokenize():
    prep = Preprocessor()

    tokens = prep.tokenize("hello world")
    assert tokens == ["hello", "world"]

    tokens = prep.tokenize("The astronauts are running")
    assert len(tokens) == 4
    print("✅ tokenize: OK")


def test_is_valid_token():
    prep = Preprocessor()

    assert prep.is_valid_token("hello")
    assert prep.is_valid_token("world")
    assert not prep.is_valid_token("123")
    assert not prep.is_valid_token("hello123")
    assert not prep.is_valid_token("!@#")
    print("✅ is_valid_token: OK")


def test_is_stopword():
    prep = Preprocessor()

    assert prep.is_stopword("the")
    assert prep.is_stopword("is")
    assert prep.is_stopword("a")
    assert not prep.is_stopword("astronaut")
    assert not prep.is_stopword("moon")
    print("✅ is_stopword: OK")


def test_stem():
    prep = Preprocessor()

    assert prep.stem("running") == "run"
    assert prep.stem("astronauts") == "astronaut"
    assert prep.stem("experiments") == "experi"
    print("✅ stem: OK")


def test_process():
    prep = Preprocessor()

    text = "The astronauts are running experiments on the Moon"
    tokens = prep.process(text)

    # Debe filtrar stopwords: the, are, on, the
    terms = [t.term for t in tokens]
    assert "the" not in terms
    assert "are" not in terms
    assert "astronaut" in terms
    assert "run" in terms
    assert "moon" in terms

    # Verificar que son objetos Token con posición
    assert all(isinstance(t, Token) for t in tokens)
    assert all(hasattr(t, "position") for t in tokens)
    print("✅ process: OK")


def test_process_to_terms():
    prep = Preprocessor()

    text = "The astronauts are running experiments on the Moon"
    terms = prep.process_to_terms(text)

    assert isinstance(terms, list)
    assert all(isinstance(t, str) for t in terms)
    assert "astronaut" in terms
    assert "run" in terms
    print("✅ process_to_terms: OK")


def test_get_term_positions():
    prep = Preprocessor()

    text = "Moon landing on the Moon surface"
    positions = prep.get_term_positions(text)

    # "moon" aparece 2 veces
    assert "moon" in positions
    assert len(positions["moon"]) == 2
    print("✅ get_term_positions: OK")


def test_get_preprocessor_singleton():
    prep1 = get_preprocessor()
    prep2 = get_preprocessor()

    assert prep1 is prep2
    print("✅ get_preprocessor singleton: OK")


def test_empty_text():
    prep = Preprocessor()

    assert prep.normalize("") == ""
    assert prep.tokenize("") == []
    assert prep.process("") == []
    assert prep.process_to_terms("") == []
    assert prep.get_term_positions("") == {}
    print("✅ empty_text: OK")


def test_punctuation_removal():
    prep = Preprocessor()

    text = "Hello, world! How are you?"
    terms = prep.process_to_terms(text)

    # No debe haber signos de puntuación
    assert "," not in terms
    assert "!" not in terms
    assert "?" not in terms
    print("✅ punctuation_removal: OK")


def test_numbers_removal():
    prep = Preprocessor()

    text = "In 2024 NASA launched Apollo 13"
    terms = prep.process_to_terms(text)

    # No debe haber números
    assert "2024" not in terms
    assert "13" not in terms
    assert "nasa" in terms
    assert "apollo" in terms
    print("✅ numbers_removal: OK")


def test_process_without_stopword_removal():
    prep = Preprocessor()

    text = "The Moon is beautiful"

    # Con stopwords removidos (default)
    terms_filtered = prep.process_to_terms(text, remove_stopwords=True)
    assert "the" not in terms_filtered
    assert "is" not in terms_filtered

    # Sin remover stopwords
    terms_all = prep.process_to_terms(text, remove_stopwords=False)
    assert "the" in terms_all
    assert "is" in terms_all
    print("✅ process_without_stopword_removal: OK")


def test_position_tracking():
    prep = Preprocessor()

    text = "astronaut moon astronaut"
    tokens = prep.process(text)

    # Verificar posiciones
    # positions = {t.term: t.position for t in tokens}
    assert tokens[0].position == 0  # astronaut
    assert tokens[1].position == 1  # moon
    assert tokens[2].position == 2  # astronaut (segunda vez)
    print("✅ position_tracking: OK")


if __name__ == "__main__":
    print("\n🧪 Testing Preprocessor\n" + "=" * 40)

    test_normalize()
    test_tokenize()
    test_is_valid_token()
    test_is_stopword()
    test_stem()
    test_process()
    test_process_to_terms()
    test_get_term_positions()
    test_get_preprocessor_singleton()
    test_empty_text()
    test_punctuation_removal()
    test_numbers_removal()
    test_process_without_stopword_removal()
    test_position_tracking()

    print("\n" + "=" * 40)
    print("✅ Todos los tests pasaron!")
