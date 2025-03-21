import faiss
import numpy as np
from models.ast_processing import parse_ast, vectorize_ast

def build_faiss_index(ast_data):
    """
    FAISS 벡터DB 생성 함수
    """
    vectors, vectorizer = vectorize_ast(ast_data)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index, vectorizer

class VectorDB:
    def __init__(self, index, vectorizer, correct_codes):
        self.index = index
        self.vectorizer = vectorizer
        self.correct_codes = correct_codes

    def get_similar_code(self, user_code):
        """
        사용자의 코드와 가장 유사한 정답 코드 검색
        """
        user_ast = parse_ast(user_code)
        user_vector = self.vectorizer.transform([user_ast]).toarray()
        _, indices = self.index.search(user_vector, k=1)
        return self.correct_codes.iloc[indices[0][0]]["code"]
