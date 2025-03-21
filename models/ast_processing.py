import ast
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_ast(code):
    """Python 코드를 AST(Abstract Syntax Tree)로 변환하여 문자열로 반환"""
    try:
        tree = ast.parse(code)
        return ast.dump(tree, indent=4)
    except Exception:
        return None  # AST 변환 실패 시 None 반환

def vectorize_ast(ast_data):
    """AST 데이터를 벡터화하는 함수"""
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(ast_data).toarray(), vectorizer
