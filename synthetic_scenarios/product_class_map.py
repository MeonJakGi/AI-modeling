# ============================================================
# product_class_map.py
# 개별상품 YOLO 모델 기준 class_id 매핑
# - JSON 내부 class_id 사용 금지
# - product_name을 YOLO 모델 class_id로 재매핑할 때 사용
# ============================================================

import unicodedata


PRODUCT_CLASS_NAMES = {
    0: "(주)국모싸이언스)메디안칼슘치약75G",
    1: "CJ다담떡볶이양념",
    2: "CJ렛츠웰맛밤80G",
    3: "CJ맥스봉체다치즈어랏400G",
    4: "CJ비비고스팸부대찌개460G",
    5: "CJ스팸200G",
    6: "CJ햇반컵반설렁탕밥253g",
    7: "골드)황도슬라이스",
    8: "길림양행)구운아몬드",
    9: "깨끗한나라여행용티슈핑크50매",
    10: "꼬깔콘고소한맛72G",
    11: "농심)사리곰탕110G(봉지)",
    12: "농심)신라면건면(낱개)97G",
    13: "농심)프링글스클래식110G",
    14: "농심매운새우깡90G",
    15: "농심보노콘스프3입18.6g",
    16: "농심새우탕컵(소)67G",
    17: "농심순한너구리120G",
    18: "농심양파링84G",
    19: "농심짜파게티범벅70G",
    20: "농심짜파게티큰사발123G",
    21: "델몬트후레쉬컷슬라이스파인애플836G",
    22: "동서맥심카누마일드로스트스위트아메리카노10T",
    23: "동아제약)가그린후레쉬라임100ML",
    24: "동원)고추참치",
    25: "동원살코기참치250G",
    26: "롯데런천미트340G",
    27: "롯데빈츠204G",
    28: "롯데칸쵸컵88G",
    29: "리뉴후래쉬60ML-1118입고",
    30: "머거본)콘소메맛아몬드",
    31: "명성식품)한입부산어포",
    32: "목우촌육포",
    33: "미소)초이스엘참치_닭가슴살5개입",
    34: "삼양)불닭소스200G",
    35: "삼양크림까르보불닭볶음면큰컵120G",
    36: "세계식품)칼몬드",
    37: "씨제이올리브영필리밀리원형구름면봉140개",
    38: "씨제이제일제당)비비고소고기죽",
    39: "에프킬라무향수성",
    40: "오뚜기3분쇠고기짜장200G",
    41: "오뚜기3분카레약간매운맛200G",
    42: "오뚜기라면볶이120G",
    43: "오뚜기스위트콘340G",
    44: "오뚜기육개장컵110G",
    45: "오뚜기전복죽285g",
    46: "오뚜기진라면매운맛(봉지)120G",
    47: "오뚜기참깨라면(컵)",
    48: "오뚜기컵밥불닭마요덮밥277g",
    49: "오뚜기튀김우동컵110G",
    50: "오리온)포카칩오리지널66G",
    51: "오리온닥터유다이제194G",
    52: "유한좋은느낌오버나이트33CM",
    53: "유한킴벌리크리넥스수앤수실키소프트라이언20매",
    54: "존슨앤드존슨)리스테린액쿨민트마일드100ML",
    55: "크라운)콘초66G",
    56: "태광푸드)볶은통귀리",
    57: "팔도더왕뚜껑컵순한맛101G",
    58: "팔도비빔면",
    59: "페브리즈에어바닐라라벤더(275ML)",
    60: "프레스코후르츠칵테일410G",
    61: "한진식품)꼬마꾸이킬(오리지널)",
    62: "한진식품)한입사각어포(매콤한맛)",
    63: "해브잇올모블프리미엄케이블C타입",
    64: "해브잇올모블프리미엄케이블애플8핀",
    65: "해태)포키46G",
    66: "해태오예스360G",
}


def normalize_product_name(name):
    """
    상품명 매칭용 정규화 함수.

    합성 JSON 상품명이 한글 자모 분리 형태여도
    YOLO 모델 class name과 매칭되도록 보정한다.
    """
    if name is None:
        return None

    name = unicodedata.normalize("NFKC", str(name)).strip()
    name = name.replace(" ", "")
    return name


PRODUCT_NAME_TO_CLASS_ID = {
    normalize_product_name(name): class_id
    for class_id, name in PRODUCT_CLASS_NAMES.items()
}


CLASS_ID_TO_PRODUCT_NAME = PRODUCT_CLASS_NAMES


def get_product_class_id(product_name, default=None):
    """
    상품명을 YOLO 모델 기준 class_id로 변환한다.

    Parameters
    ----------
    product_name : str
        JSON object의 product_name / actual_product_name / expected_product_name 등
    default : any
        매칭 실패 시 반환할 값

    Returns
    -------
    int or default
        YOLO 모델 기준 class_id
    """
    key = normalize_product_name(product_name)
    return PRODUCT_NAME_TO_CLASS_ID.get(key, default)


def get_product_name_from_obj(obj):
    """
    합성 JSON object에서 상품명을 안전하게 가져온다.
    """
    return (
        obj.get("product_name")
        or obj.get("actual_product_name")
        or obj.get("target_product_name")
        or obj.get("expected_product_name")
    )


def get_product_class_id_from_obj(obj, default=None):
    """
    합성 JSON object에서 상품명을 꺼내 YOLO 모델 기준 class_id로 변환한다.

    주의: JSON 내부 class_id는 사용하지 않는다.
    """
    product_name = get_product_name_from_obj(obj)
    return get_product_class_id(product_name, default=default)


def validate_class_map():
    """
    class map 기본 검수.
    """
    class_ids = sorted(PRODUCT_CLASS_NAMES.keys())

    assert len(PRODUCT_CLASS_NAMES) == 67, "상품 class 수가 67개가 아닙니다."
    assert class_ids[0] == 0, "class_id가 0부터 시작하지 않습니다."
    assert class_ids[-1] == 66, "class_id가 66까지 있지 않습니다."
    assert len(PRODUCT_NAME_TO_CLASS_ID) == 67, "상품명 매핑 개수가 67개가 아닙니다."

    return {
        "num_classes": len(PRODUCT_CLASS_NAMES),
        "min_class_id": class_ids[0],
        "max_class_id": class_ids[-1],
        "num_name_map": len(PRODUCT_NAME_TO_CLASS_ID),
    }
