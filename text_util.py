import re

# 한글 범위의 정규 표현식 패턴
korean_pattern = re.compile('[^ ㄱ-ㅣ가-힣]+')


def extract_korean(text):
    # 정규 표현식을 사용하여 한글만 추출
    korean_only = korean_pattern.sub('', text)

    return korean_only