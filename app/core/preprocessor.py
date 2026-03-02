import re
from dataclasses import dataclass
from typing import List
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


@dataclass
class Token:
    term: str
    position: int


class Preprocessor:
    def __init__(self, language: str = "english"):
        self.language = language
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words(language))
        self.word_pattern = re.compile(r"^[a-zA-Z]+$")

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        return text.lower()

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return word_tokenize(text)

    def is_valid_token(self, token: str) -> bool:
        return bool(self.word_pattern.match(token))

    def is_stopword(self, token: str) -> bool:
        return token in self.stop_words

    def stem(self, token: str) -> str:
        return self.stemmer.stem(token)

    def process(self, text: str, remove_stopwords: bool = True) -> List[Token]:
        tokens = []
        normalized_text = self.normalize(text)
        raw_tokens = self.tokenize(normalized_text)

        position = 0
        for raw_token in raw_tokens:
            if not self.is_valid_token(raw_token):
                continue

            if remove_stopwords and self.is_stopword(raw_token):
                position += 1
                continue

            term = self.stem(raw_token)
            token = Token(term=term, position=position)
            tokens.append(token)
            position += 1

        return tokens

    def process_to_terms(self, text: str, remove_stopwords: bool = True) -> List[str]:
        tokens = self.process(text, remove_stopwords)
        return [token.term for token in tokens]

    def get_term_positions(self, text: str, remove_stopwords: bool = True) -> dict:
        tokens = self.process(text, remove_stopwords)
        term_positions = {}

        for token in tokens:
            if token.term not in term_positions:
                term_positions[token.term] = []
            term_positions[token.term].append(token.position)

        return term_positions


_preprocessor = None


def get_preprocessor() -> Preprocessor:
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = Preprocessor()
    return _preprocessor
