import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import faiss
import numpy as np

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI LLM 설정
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

# 🚀 미완성 코드 프롬프트
prompt_incomplete_code = PromptTemplate(
    input_variables=["user_code"],
    template="""사용자가 다음 코드를 작성하다가 중간에 막혔습니다:

    ```python
    {user_code}
    ```

    1️⃣ 사용자가 어떤 부분에서 막혔을 가능성이 높은지 분석하세요.
    2️⃣ 사용자가 스스로 해결할 수 있도록 단계별 힌트를 제공하세요.
    3️⃣ 정답을 직접 주지 말고, 사고 과정을 확장할 수 있도록 질문을 던지세요.
    """
)

# 🚀 완성된 코드 프롬프트
prompt_complete_code = PromptTemplate(
    input_variables=["user_code"],
    template="""사용자가 다음 코드를 작성했습니다:

    ```python
    {user_code}
    ```

    1️⃣ 이 코드가 어떤 알고리즘을 사용하고 있는지 설명하세요.
    2️⃣ 논리적 오류가 있는지 분석하고, 오류가 있다면 문제점을 설명하세요.
    3️⃣ 코드의 성능을 최적화할 수 있는 방법을 유도 질문 형식으로 제시하세요.
    4️⃣ 사용자가 직접 생각할 수 있도록 정답을 주지 말고, 최적화 방향을 제안하세요.
    """
)

# LangChain 체인 설정
chain_incomplete = LLMChain(llm=llm, prompt=prompt_incomplete_code)
chain_complete = LLMChain(llm=llm, prompt=prompt_complete_code)

def analyze_code(user_code, vector_db=None):
    """
    사용자의 코드가 미완성인지 완성된 코드인지 판별하고, AI 분석을 실행하는 함수.
    미완성 코드라면 RAG 기반 힌트를 제공.
    """
    if user_code.strip().endswith(":") or "pass" in user_code:
        response = chain_incomplete.run(user_code=user_code)
        # RAG를 활용하여 미완성 코드에 대해 유사한 예시 코드 제공
        if vector_db:
            similar_code = vector_db.get_similar_code(user_code)
            response += f"\n📌 유사한 코드 예시:\n```python\n{similar_code}\n```"
        return response
    else:
        return chain_complete.run(user_code=user_code)
