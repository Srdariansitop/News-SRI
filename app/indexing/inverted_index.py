import json
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class Posting:
    doc_id: str
    tf: int = 1
    positions: List[int] = field(default_factory=list)


@dataclass
class TermEntry:
    df: int = 0
    postings: List[Posting] = field(default_factory=list)


class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, TermEntry] = {}
        self.doc_count: int = 0
        self.doc_lengths: Dict[str, int] = {}

    def add_document(self, doc_id: str, term_positions: Dict[str, List[int]]):
        doc_length = 0

        for term, positions in term_positions.items():
            tf = len(positions)
            doc_length += tf

            if term not in self.index:
                self.index[term] = TermEntry()

            entry = self.index[term]
            entry.df += 1
            entry.postings.append(Posting(doc_id=doc_id, tf=tf, positions=positions))

        self.doc_lengths[doc_id] = doc_length
        self.doc_count += 1

    def get_postings(self, term: str) -> List[Posting]:
        if term in self.index:
            return self.index[term].postings
        return []

    def get_document_frequency(self, term: str) -> int:
        if term in self.index:
            return self.index[term].df
        return 0

    def get_term_frequency(self, term: str, doc_id: str) -> int:
        for posting in self.get_postings(term):
            if posting.doc_id == doc_id:
                return posting.tf
        return 0

    def get_vocabulary(self) -> List[str]:
        return list(self.index.keys())

    def get_vocabulary_size(self) -> int:
        return len(self.index)

    def get_average_doc_length(self) -> float:
        if self.doc_count == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / self.doc_count

    def contains_term(self, term: str) -> bool:
        return term in self.index

    def get_docs_containing_term(self, term: str) -> List[str]:
        return [p.doc_id for p in self.get_postings(term)]

    def save(self, filepath: str):
        data = {
            "doc_count": self.doc_count,
            "doc_lengths": self.doc_lengths,
            "index": {
                term: {
                    "df": entry.df,
                    "postings": [
                        {"doc_id": p.doc_id, "tf": p.tf, "positions": p.positions}
                        for p in entry.postings
                    ],
                }
                for term, entry in self.index.items()
            },
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.doc_count = data["doc_count"]
        self.doc_lengths = data["doc_lengths"]
        self.index = {}

        for term, entry_data in data["index"].items():
            postings = [
                Posting(doc_id=p["doc_id"], tf=p["tf"], positions=p["positions"])
                for p in entry_data["postings"]
            ]
            self.index[term] = TermEntry(df=entry_data["df"], postings=postings)

    def clear(self):
        self.index.clear()
        self.doc_lengths.clear()
        self.doc_count = 0
