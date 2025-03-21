import pandas as pd

def load_and_preprocess_data(file_paths):
    """
    CSV 파일을 로드하고, 정답 코드와 오류 코드를 분리하는 함수.
    """
    df_list = [pd.read_csv(file) for file in file_paths]
    df = pd.concat(df_list, ignore_index=True)
    
    # 필요한 컬럼만 유지
    columns_to_keep = ["problem_id", "problem_title", "category", "result", "language", "code"]
    df_cleaned = df[columns_to_keep].dropna()

    # 정답 코드 & 오류 코드 분리
    df_correct = df_cleaned[df_cleaned["result"] == "맞았습니다!!"]
    df_error = df_cleaned[df_cleaned["result"] != "맞았습니다!!"]

    print(f"✅ 데이터 로드 완료: 정답 코드 {len(df_correct)}개, 오류 코드 {len(df_error)}개")
    
    return df_correct, df_error

# 사용 예제
if __name__ == "__main__":
    file_paths = ["./data/code_data-1.csv", "./data/code_data-2.csv"]
    load_and_preprocess_data(file_paths)
