"""
shelf_synthetic_common.py

합성 데이터 노트북에서 공통으로 쓰는 환경/유틸 함수 모음입니다.

중요:
- 이 파일은 S2/S3/S4/S5 같은 시나리오별 로직을 만들지 않습니다.
- 이 파일은 시나리오별 저장 경로를 만들거나 결과 파일을 저장하지 않습니다.
- 각 시나리오의 조건 설정, output 폴더 생성, 이미지/YOLO txt 저장은 ipynb에서 직접 수행합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import random
import re

from PIL import Image, ImageEnhance, ImageFilter, ImageDraw


# -----------------------------------------------------------------------------
# 1. 공통 고정값
# -----------------------------------------------------------------------------

FIXED_DISPLAY_QTY_BY_PRODUCT_ID: Dict[str, int] = {
    "10210": 6,   # 포카칩
    "20126": 8,   # 프링글스
    "80092": 6,   # 포키
    "10093": 4,   # 새우깡
    "10091": 5,   # 꼬깔콘
    "90089": 6,   # 칸쵸
    "90125": 4,   # 양파링
    "20141": 4,   # 오예스
    "55029": 2,   # 빈츠
    "10094": 4,   # 콘초

    "60121": 4,   # 튀김우동
    "15833": 4,   # 더왕뚜껑
    "40139": 4,   # 사리곰탕
    "30123": 4,   # 신라면 건면
    "60114": 4,   # 짜파게티
    "70138": 4,   # 너구리
    "15838": 4,   # 육개장
    "20113": 4,   # 짜파게티
    "15839": 4,   # 불닭
    "90142": 2,   # 진라면
    "20112": 4,   # 라면볶이
    "70136": 3,   # 팔도비빔면
    "10103": 4,   # 참깨라면
}

# 선반 위치 좌표: 한 행당 상품 시작점 4개 + 행 끝점 1개 = 총 5개
ROW_POINTS: List[Tuple[int, int]] = [
    (312, 455), (970, 452), (1711, 444), (2377, 461), (3032, 463),
    (346, 695), (992, 692), (1728, 692), (2391, 703), (3001, 706),
    (382, 959), (992, 948), (1728, 951), (2391, 948), (2951, 948),
    (424, 1193), (1020, 1193), (1725, 1182), (2363, 1191), (2901, 1199),
    (460, 1436), (1034, 1430), (1717, 1436), (2363, 1428), (2865, 1439),
]

SHELF_LIP_POINTS: List[Tuple[int, int]] = [
    (281, 466), (3051, 480), (3054, 530), (279, 525),
    (326, 723), (3012, 720), (3007, 770), (329, 773),
    (371, 971), (2965, 968), (2959, 1015), (373, 1015),
    (410, 1210), (2906, 1213), (2906, 1258), (410, 1263),
    (454, 1453), (2865, 1447), (2865, 1497), (451, 1500),
]

PRODUCTS_PER_ROW = 4
POINTS_PER_ROW = PRODUCTS_PER_ROW + 1
POINTS_PER_LIP = 4
NUM_LIPS = 5

# 현재 좌표 기준: 1~2열 과자, 3~4열 면류
CATEGORY_BY_COL = {
    1: ("snack", "과자"),
    2: ("snack", "과자"),
    3: ("noodle", "면류"),
    4: ("noodle", "면류"),
}


# -----------------------------------------------------------------------------
# 2. 경로 / 데이터 로딩
# -----------------------------------------------------------------------------

def make_default_paths(base_dir: Path, background_name: str = "선반이미지_정면.png") -> Dict[str, Path]:
    """프로젝트 폴더 구조에 맞춰 기본 경로만 만든다. 폴더 생성은 하지 않는다."""
    base_dir = Path(base_dir)
    return {
        "base_dir": base_dir,
        "background_path": base_dir / "창고 이미지" / background_name,
        "snack_fg_dir": base_dir / "dataset" / "bg_removed" / "과자",
        "noodle_fg_dir": base_dir / "dataset" / "bg_removed" / "면류",
        "synthetic_root_dir": base_dir / "dataset" / "synthetic",
    }


def validate_paths(paths: Dict[str, Path]) -> None:
    """필수 경로 존재 여부를 확인한다."""
    required_keys = ["background_path", "snack_fg_dir", "noodle_fg_dir"]
    missing = [key for key in required_keys if not paths[key].exists()]

    if missing:
        message = "\n".join([f"- {key}: {paths[key]}" for key in missing])
        raise FileNotFoundError(
            "필수 경로가 존재하지 않습니다. BASE_DIR 또는 폴더명을 확인하세요.\n" + message
        )


def list_product_dirs(fg_dir: Path) -> List[Path]:
    """bg_removed 하위에서 output 폴더가 있는 상품 폴더만 가져온다."""
    fg_dir = Path(fg_dir)
    return sorted([p for p in fg_dir.iterdir() if p.is_dir() and (p / "output").exists()])


def get_product_info_from_dir(product_dir: Path) -> Tuple[str, str]:
    """상품 폴더명에서 product_id, product_name을 분리한다."""
    folder_name = Path(product_dir).name
    if "_" in folder_name:
        product_id, product_name = folder_name.split("_", 1)
    else:
        product_id = folder_name
        product_name = folder_name
    return str(product_id), product_name


def get_fixed_display_qty(product_id: str) -> int:
    """상품 ID 기준 앞줄 기준 진열 개수를 가져온다."""
    product_id = str(product_id)
    if product_id not in FIXED_DISPLAY_QTY_BY_PRODUCT_ID:
        print("[수량 정보 없음] 기본값 1개 사용:", product_id)
        return 1
    return FIXED_DISPLAY_QTY_BY_PRODUCT_ID[product_id]


def parse_png_info(png_path: Path) -> Optional[Dict[str, Any]]:
    """png 파일명에서 product_id, pitch, image_type, roll_no를 파싱한다."""
    name = Path(png_path).name
    match = re.search(r"^(.+?)_(\d+)_(m|s)_(\d+)_", name)
    if match is None:
        return None

    return {
        "product_id": match.group(1),
        "pitch": int(match.group(2)),
        "image_type": match.group(3),
        "roll_no": int(match.group(4)),
        "filename": name,
    }


# -----------------------------------------------------------------------------
# 3. 슬롯 / 선반 앞턱 생성
# -----------------------------------------------------------------------------

def make_slots_from_row_points(
    row_points: List[Tuple[int, int]] = ROW_POINTS,
    slot_h: int = 190,
    right_margin: int = 30,
    left_margin: int = 40,
) -> List[Dict[str, Any]]:
    """클릭 좌표 기반으로 각 상품 슬롯 bbox를 생성한다."""
    all_slots: List[Dict[str, Any]] = []
    snack_no = 1
    noodle_no = 1

    num_rows = len(row_points) // POINTS_PER_ROW

    for row_idx in range(num_rows):
        row_no = row_idx + 1
        start = row_idx * POINTS_PER_ROW
        end = start + POINTS_PER_ROW
        points_this_row = row_points[start:end]

        product_points = points_this_row[:PRODUCTS_PER_ROW]
        row_end_x, _ = points_this_row[-1]

        for col_idx, (ax, ay) in enumerate(product_points, start=1):
            zone_id, category = CATEGORY_BY_COL[col_idx]

            if col_idx < PRODUCTS_PER_ROW:
                next_x, _ = product_points[col_idx]
                x2 = next_x - right_margin
            else:
                x2 = row_end_x - right_margin

            if zone_id == "snack":
                product_no = snack_no
                snack_no += 1
            else:
                product_no = noodle_no
                noodle_no += 1

            all_slots.append({
                "slot_id": f"{zone_id}-{product_no:02d}",
                "zone_id": zone_id,
                "category": category,
                "product_no": product_no,
                "row_no": row_no,
                "col_no": col_idx,
                "anchor_x": int(ax),
                "anchor_y": int(ay),
                "x1": int(ax - left_margin),
                "y1": int(ay - slot_h),
                "x2": int(x2),
                "y2": int(ay),
                "cx": int(ax),
                "cy": int(ay),
            })

    return all_slots


def make_shelf_lip_polygons(shelf_lip_points: List[Tuple[int, int]] = SHELF_LIP_POINTS) -> List[List[Tuple[int, int]]]:
    """선반 앞턱 좌표를 4점 polygon 단위로 묶는다."""
    polygons: List[List[Tuple[int, int]]] = []
    for i in range(NUM_LIPS):
        start = i * POINTS_PER_LIP
        end = start + POINTS_PER_LIP
        polygons.append(shelf_lip_points[start:end])
    return polygons


def create_occluder_from_polygons(background_rgba: Image.Image, polygons: List[List[Tuple[int, int]]]) -> Image.Image:
    """선반 앞턱 부분만 배경에서 잘라낸 occluder를 만든다."""
    w, h = background_rgba.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for poly in polygons:
        draw.polygon(poly, fill=255)

    occluder = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    occluder.paste(background_rgba, (0, 0), mask)
    return occluder


# -----------------------------------------------------------------------------
# 4. 공통 컨텍스트 생성
# -----------------------------------------------------------------------------

def make_shuffled_product_map(product_dirs: List[Path], slots: List[Dict[str, Any]], seed: Optional[int] = None) -> Dict[int, Path]:
    """상품 목록을 섞어서 슬롯 product_no에 매핑한다."""
    if seed is not None:
        random.seed(seed)

    product_dirs = list(product_dirs)
    if not product_dirs:
        raise ValueError("상품 폴더가 비어 있습니다.")

    random.shuffle(product_dirs)

    if len(product_dirs) >= len(slots):
        selected_products = product_dirs[:len(slots)]
    else:
        extra = random.choices(product_dirs, k=len(slots) - len(product_dirs))
        selected_products = product_dirs + extra

    return {slot["product_no"]: product_dir for slot, product_dir in zip(slots, selected_products)}


def build_product_class_map(snack_product_dirs: List[Path], noodle_product_dirs: List[Path]) -> Dict[str, int]:
    """상품 ID 기준 YOLO class map을 만든다."""
    product_ids: List[str] = []
    for product_dir in list(snack_product_dirs) + list(noodle_product_dirs):
        product_id, _ = get_product_info_from_dir(product_dir)
        product_ids.append(str(product_id))

    product_ids = sorted(set(product_ids))
    return {product_id: class_id for class_id, product_id in enumerate(product_ids)}


def create_synthetic_context(
    base_dir: Path,
    background_name: str = "선반이미지_정면.png",
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    합성에 필요한 공통 객체만 준비한다.

    반환값에는 배경, 상품 폴더 목록, 슬롯, product_map, occluder, class_map이 들어간다.
    저장 경로 생성과 파일 저장은 각 ipynb에서 수행한다.
    """
    if seed is not None:
        random.seed(seed)

    paths = make_default_paths(Path(base_dir), background_name=background_name)
    validate_paths(paths)

    background = Image.open(paths["background_path"]).convert("RGB")

    snack_product_dirs = list_product_dirs(paths["snack_fg_dir"])
    noodle_product_dirs = list_product_dirs(paths["noodle_fg_dir"])

    all_slots = make_slots_from_row_points(ROW_POINTS)
    snack_slots = [slot for slot in all_slots if slot["zone_id"] == "snack"]
    noodle_slots = [slot for slot in all_slots if slot["zone_id"] == "noodle"]

    snack_product_map = make_shuffled_product_map(snack_product_dirs, snack_slots, seed=seed)
    noodle_product_map = make_shuffled_product_map(noodle_product_dirs, noodle_slots, seed=None if seed is None else seed + 1)

    all_zones = [
        {
            "zone_id": "snack",
            "category": "과자",
            "slots": snack_slots,
            "product_map": snack_product_map,
        },
        {
            "zone_id": "noodle",
            "category": "면류",
            "slots": noodle_slots,
            "product_map": noodle_product_map,
        },
    ]

    shelf_lip_polygons = make_shelf_lip_polygons(SHELF_LIP_POINTS)
    combined_occluder = create_occluder_from_polygons(background.convert("RGBA"), shelf_lip_polygons)
    class_map = build_product_class_map(snack_product_dirs, noodle_product_dirs)

    return {
        "paths": paths,
        "background": background,
        "snack_product_dirs": snack_product_dirs,
        "noodle_product_dirs": noodle_product_dirs,
        "all_slots": all_slots,
        "snack_slots": snack_slots,
        "noodle_slots": noodle_slots,
        "all_zones": all_zones,
        "shelf_lip_polygons": shelf_lip_polygons,
        "combined_occluder": combined_occluder,
        "class_map": class_map,
    }


def collect_slot_items(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """context 안의 slot과 기준 상품 정보를 한 줄씩 펼친다."""
    items: List[Dict[str, Any]] = []

    for zone in ctx["all_zones"]:
        zone_id = zone["zone_id"]
        category = zone["category"]
        slots = zone["slots"]
        product_map = zone["product_map"]

        for slot in slots:
            product_no = slot["product_no"]
            if product_no not in product_map:
                continue

            product_dir = product_map[product_no]
            product_id, product_name = get_product_info_from_dir(product_dir)

            items.append({
                "zone_id": zone_id,
                "category": category,
                "slot": slot,
                "product_no": product_no,
                "product_dir": product_dir,
                "product_id": str(product_id),
                "product_name": product_name,
            })

    return items


def make_normal_slot_plan(ctx: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    모든 슬롯을 정상 진열 상태로 둔 기본 plan을 만든다.
    각 ipynb에서 필요한 슬롯만 수정해 S2/S3/S4/S5 조건을 만들면 된다.
    """
    plan: Dict[str, Dict[str, Any]] = {}

    for item in collect_slot_items(ctx):
        slot = item["slot"]
        slot_id = slot["slot_id"]
        product_id = item["product_id"]
        product_name = item["product_name"]
        normal_qty = get_fixed_display_qty(product_id)

        plan[slot_id] = {
            "slot": slot,
            "zone_id": item["zone_id"],
            "category": item["category"],
            "product_no": item["product_no"],

            "target_product_dir": item["product_dir"],
            "target_product_id": product_id,
            "target_product_name": product_name,
            "normal_front_qty": normal_qty,
            "normal_back_qty": normal_qty,

            "actual_product_dir": item["product_dir"],
            "actual_product_id": product_id,
            "actual_product_name": product_name,
            "display_qty": normal_qty,
            "back_display_qty": normal_qty,

            "front_missing_indices": [],
            "back_missing_indices": [],
            "back_visible_indices": None,

            # 아래 필드는 ipynb에서 시나리오에 맞게 덮어쓴다.
            "scenario_code": "S0",
            "action": "normal",
            "list_up": False,
            "final_status": "정상",
            "missing_qty": 0,
            "target_column_index": None,
            "is_misplaced": False,
        }

    return plan


def choose_target_slot(plan: Dict[str, Dict[str, Any]], target_slot_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """target_slot_id가 없으면 plan 안에서 랜덤으로 하나를 고른다."""
    if target_slot_id is None:
        slot_id = random.choice(list(plan.keys()))
        return slot_id, plan[slot_id]

    if target_slot_id not in plan:
        valid_ids = list(plan.keys())
        raise ValueError(f"target_slot_id를 찾을 수 없습니다: {target_slot_id}\n가능한 slot_id 예시: {valid_ids[:10]}")

    return target_slot_id, plan[target_slot_id]


# -----------------------------------------------------------------------------
# 5. 상품 이미지 합성 유틸
# -----------------------------------------------------------------------------

def get_scale_by_slot(slot: Dict[str, Any]) -> float:
    """슬롯별 상품 크기 조절값. 필요하면 row_no/zone_id별로 분기 가능."""
    return 0.95


def adjust_product_color(product: Image.Image) -> Image.Image:
    """선반 조명에 맞게 상품 색/선명도를 살짝 조정한다."""
    product = ImageEnhance.Brightness(product).enhance(random.uniform(0.92, 1.02))
    product = ImageEnhance.Contrast(product).enhance(random.uniform(0.92, 1.00))
    product = ImageEnhance.Color(product).enhance(random.uniform(0.90, 1.00))
    product = ImageEnhance.Sharpness(product).enhance(random.uniform(0.90, 1.00))
    product = product.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.05, 0.15)))
    return product


def add_shadow(
    base: Image.Image,
    product: Image.Image,
    paste_x: int,
    paste_y: int,
    offset: Tuple[int, int] = (5, 8),
    blur: int = 8,
    opacity: int = 45,
) -> None:
    """상품 알파 영역을 이용해 그림자를 만든다."""
    alpha = product.getchannel("A")
    shadow = Image.new("RGBA", product.size, (0, 0, 0, opacity))
    shadow.putalpha(alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    base.alpha_composite(shadow, (paste_x + offset[0], paste_y + offset[1]))


def get_png_by_slot_view(product_dir: Path, slot: Dict[str, Any], image_type: str = "s") -> Optional[Path]:
    """슬롯 층에 맞는 상품 PNG를 선택한다."""
    output_dir = Path(product_dir) / "output"
    png_paths = list(output_dir.glob("*.png"))

    if len(png_paths) == 0:
        print("[PNG 없음]", product_dir)
        return None

    if slot["row_no"] == 1:
        target_patterns = [f"_60_{image_type}_1_", f"_60_{image_type}_2_", f"_60_{image_type}_3_"]
    else:
        target_patterns = [f"_30_{image_type}_1_", f"_30_{image_type}_2_", f"_30_{image_type}_3_"]

    for pattern in target_patterns:
        candidates = [p for p in png_paths if pattern in p.name]
        if len(candidates) > 0:
            return random.choice(candidates)

    print(
        "[1,2,3 각도 파일 모두 없음]",
        "slot:", slot["slot_id"],
        "row_no:", slot["row_no"],
        "product_dir:", Path(product_dir).name,
        "target_patterns:", target_patterns,
    )
    return None


def paste_products_in_slot(
    result: Image.Image,
    product_dir: Path,
    slot: Dict[str, Any],
    display_qty: Optional[int] = None,
    back_display_qty: Optional[int] = None,
    front_missing_indices: Optional[List[int]] = None,
    back_missing_indices: Optional[List[int]] = None,
    back_visible_indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """하나의 slot에 앞줄/뒷줄 상품을 붙인다."""
    front_missing_set = set(front_missing_indices or [])
    back_missing_set = set(back_missing_indices or [])
    back_visible_set = set(back_visible_indices) if back_visible_indices is not None else None

    anchor_x = slot["anchor_x"]
    anchor_y = slot["anchor_y"]

    base_product_path = get_png_by_slot_view(product_dir=product_dir, slot=slot, image_type="s")
    if base_product_path is None:
        return []

    base_product = Image.open(base_product_path).convert("RGBA")

    dynamic_scale = get_scale_by_slot(slot)
    slot_height = slot["y2"] - slot["y1"]
    product_h = int(slot_height * dynamic_scale)
    ratio = product_h / base_product.height
    product_w = int(base_product.width * ratio)

    if display_qty is None:
        display_qty = 1
    if back_display_qty is None:
        back_display_qty = display_qty

    layout_qty = max(display_qty, back_display_qty)

    def calculate_center_x_list(qty: int) -> Tuple[int, int, List[float]]:
        row_product_w = int(product_w)
        row_product_h = int(product_h)

        slot_left = slot["x1"]
        slot_right = slot["x2"]

        min_center_x = slot_left + row_product_w / 2
        max_center_x = slot_right - row_product_w / 2

        if max_center_x < min_center_x:
            max_width = max(20, slot_right - slot_left)
            shrink_ratio = max_width / row_product_w * 0.95
            row_product_w = int(row_product_w * shrink_ratio)
            row_product_h = int(row_product_h * shrink_ratio)
            min_center_x = slot_left + row_product_w / 2
            max_center_x = slot_right - row_product_w / 2

        start_center_x = max(anchor_x, min_center_x)
        start_center_x = min(start_center_x, max_center_x)

        if qty <= 1:
            step_x = 0
        else:
            default_overlap_ratio = 0.15
            default_step_x = row_product_w * (1 - default_overlap_ratio)
            available_center_width = max_center_x - start_center_x
            step_x = min(default_step_x, available_center_width / (qty - 1))
            step_x = max(0, step_x)

        center_x_list = [start_center_x + i * step_x for i in range(qty)]
        return row_product_w, row_product_h, center_x_list

    front_product_w, front_product_h, center_x_list = calculate_center_x_list(layout_qty)

    def get_back_row_config(slot_: Dict[str, Any]) -> Dict[str, float]:
        if slot_["row_no"] == 1:
            return {"dy": 70, "scale": 0.88, "brightness": 0.82, "contrast": 0.90}
        return {"dy": 45, "scale": 0.90, "brightness": 0.78, "contrast": 0.88}

    def paste_one_row(
        qty: int,
        row_anchor_y: int,
        row_scale: float = 1.0,
        depth_name: str = "front",
        brightness: float = 1.0,
        contrast: float = 1.0,
    ) -> List[Dict[str, Any]]:
        row_bboxes: List[Dict[str, Any]] = []
        if qty <= 0:
            return row_bboxes

        row_product_w = int(front_product_w * row_scale)
        row_product_h = int(front_product_h * row_scale)

        for i in range(qty):
            if depth_name == "front" and i in front_missing_set:
                continue
            if depth_name == "back":
                if slot["row_no"] != 1:
                    if i in back_missing_indices:
                        continue

                    if back_visible_indices is not None:
                        if i not in back_visible_indices:
                            continue

            product = base_product.copy()
            size_jitter = random.uniform(0.98, 1.02)
            this_product_w = int(row_product_w * size_jitter)
            this_product_h = int(row_product_h * size_jitter)

            product = product.resize((this_product_w, this_product_h), resample=Image.LANCZOS)
            product = adjust_product_color(product)

            if depth_name == "back":
                product = ImageEnhance.Brightness(product).enhance(brightness)
                product = ImageEnhance.Contrast(product).enhance(contrast)

            angle = random.uniform(-0.2, 0.2)
            product = product.rotate(angle, expand=True, resample=Image.BICUBIC)

            rotated_w, rotated_h = product.size
            item_anchor_x = center_x_list[i]
            item_anchor_y = row_anchor_y

            paste_x = int(item_anchor_x - rotated_w / 2)
            paste_y = int(item_anchor_y - rotated_h)

            if paste_x < slot["x1"]:
                paste_x = slot["x1"]
            if paste_x + rotated_w > slot["x2"]:
                paste_x = slot["x2"] - rotated_w

            paste_x += random.randint(-1, 1)
            paste_y += random.randint(-1, 1)

            add_shadow(result, product, paste_x, paste_y)
            result.paste(product, (paste_x, paste_y), product)

            row_bboxes.append({
                "bbox": [paste_x, paste_y, paste_x + rotated_w, paste_y + rotated_h],
                "source_png": str(base_product_path),
                "angle_info": parse_png_info(base_product_path),
                "depth_row": depth_name,
                "position_index": i,
            })

        return row_bboxes

    bboxes: List[Dict[str, Any]] = []
    back_cfg = get_back_row_config(slot)

    # 뒷줄 먼저 붙이고, 앞줄을 나중에 붙여 앞줄이 자연스럽게 가리도록 한다.
    bboxes.extend(paste_one_row(
        qty=layout_qty,
        row_anchor_y=anchor_y - int(back_cfg["dy"]),
        row_scale=float(back_cfg["scale"]),
        depth_name="back",
        brightness=float(back_cfg["brightness"]),
        contrast=float(back_cfg["contrast"]),
    ))

    bboxes.extend(paste_one_row(
        qty=layout_qty,
        row_anchor_y=anchor_y,
        row_scale=1.0,
        depth_name="front",
    ))

    return bboxes


def render_from_slot_plan(
    ctx: Dict[str, Any],
    slot_plan: Dict[str, Dict[str, Any]],
    seed: Optional[int] = None,
) -> Tuple[Image.Image, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """ipynb에서 만든 slot_plan을 받아 합성 이미지를 생성한다."""
    if seed is not None:
        random.seed(seed)

    result = ctx["background"].convert("RGBA").copy()
    objects: List[Dict[str, Any]] = []
    slot_labels: List[Dict[str, Any]] = []
    object_id = 1

    for zone in ctx["all_zones"]:
        zone_id = zone["zone_id"]
        category = zone["category"]

        for slot in zone["slots"]:
            slot_id = slot["slot_id"]
            if slot_id not in slot_plan:
                continue

            p = slot_plan[slot_id]
            actual_product_dir = p.get("actual_product_dir")
            pasted_items: List[Dict[str, Any]] = []

            if actual_product_dir is not None and int(p.get("display_qty", 0)) > 0:
                pasted_items = paste_products_in_slot(
                    result=result,
                    product_dir=actual_product_dir,
                    slot=slot,
                    display_qty=int(p.get("display_qty", 1)),
                    back_display_qty=int(p.get("back_display_qty", p.get("display_qty", 1))),
                    front_missing_indices=p.get("front_missing_indices", []),
                    back_missing_indices=p.get("back_missing_indices", []),
                    back_visible_indices=p.get("back_visible_indices", None),
                )
            # -------------------------------------------------
            # extra_misplaced_items 처리
            # slot 전체가 아니라 특정 위치에 wrong 상품 1개만 추가로 붙임
            # -------------------------------------------------
            extra_misplaced_items = p.get("extra_misplaced_items", [])

            for extra in extra_misplaced_items:
                wrong_product_dir = extra.get("product_dir")
                wrong_product_id = extra.get("product_id")
                wrong_product_name = extra.get("product_name")
                position_index = int(extra.get("position_index", 0))

                # 핵심: 무조건 먼저 빈 리스트로 초기화
                wrong_pasted_items = []

                # product_dir가 있을 때만 wrong 상품 붙이기
                if wrong_product_dir is not None:
                    normal_qty_for_layout = int(
                        p.get("normal_front_qty", p.get("display_qty", 1))
                    )

                    # wrong 상품은 position_index 위치에만 붙이고,
                    # 나머지 위치는 전부 비움
                    wrong_front_missing_indices = [
                        i for i in range(normal_qty_for_layout)
                        if i != position_index
                    ]

                    wrong_pasted_items = paste_products_in_slot(
                        result=result,
                        product_dir=wrong_product_dir,
                        slot=slot,
                        display_qty=normal_qty_for_layout,
                        back_display_qty=0,
                        front_missing_indices=wrong_front_missing_indices,
                        back_missing_indices=list(range(normal_qty_for_layout)),
                        back_visible_indices=[],
                    )

                # 중요: 이 for문은 for extra 안에 있어야 함
                for wrong_item in wrong_pasted_items:
                    pasted_items.append({
                        **wrong_item,
                        "override_product_id": wrong_product_id,
                        "override_product_name": wrong_product_name,
                        "override_action": "misplaced_one_item",
                        "override_is_misplaced": True,
                    })

                

            front_display_qty = sum(1 for item in pasted_items if item.get("depth_row", "front") == "front")
            back_display_qty = sum(1 for item in pasted_items if item.get("depth_row") == "back")

            slot_labels.append({
                "slot_id": slot_id,
                "zone_id": zone_id,
                "category": category,
                "product_no": p.get("product_no"),
                "row_no": slot["row_no"],
                "col_no": slot["col_no"],
                "target_product_id": p.get("target_product_id"),
                "target_product_name": p.get("target_product_name"),
                "actual_product_id": p.get("actual_product_id"),
                "actual_product_name": p.get("actual_product_name"),
                "scenario_code": p.get("scenario_code"),
                "action": p.get("action"),
                "list_up": p.get("list_up", False),
                "final_status": p.get("final_status"),
                "is_misplaced": p.get("is_misplaced", False),
                "normal_front_qty": p.get("normal_front_qty"),
                "normal_back_qty": p.get("normal_back_qty"),
                "front_display_qty": front_display_qty,
                "back_display_qty": back_display_qty,
                "total_pasted_qty": len(pasted_items),
                "missing_qty": p.get("missing_qty", 0),
                "front_missing_indices": p.get("front_missing_indices", []),
                "back_missing_indices": p.get("back_missing_indices", []),
                "target_column_index": p.get("target_column_index"),
            })

            for item in pasted_items:
                item_product_id = item.get("override_product_id", p.get("actual_product_id"))
                item_product_name = item.get("override_product_name", p.get("actual_product_name"))
                item_action = item.get("override_action", p.get("action"))
                item_is_misplaced = item.get("override_is_misplaced", p.get("is_misplaced", False))

                objects.append({
                    "object_id": object_id,
                    "slot_id": slot_id,
                    "zone_id": zone_id,
                    "category": category,
                    "row_no": slot["row_no"],
                    "col_no": slot["col_no"],
                    "product_no": p.get("product_no"),

                    # 일반 상품이면 actual_product_id,
                    # 오진열 1개 상품이면 wrong_product_id가 들어감
                    "product_id": item_product_id,
                    "product_name": item_product_name,

                    "actual_product_id": p.get("actual_product_id"),
                    "actual_product_name": p.get("actual_product_name"),
                    "target_product_id": p.get("target_product_id"),
                    "target_product_name": p.get("target_product_name"),

                    "scenario_code": p.get("scenario_code"),
                    "action": item_action,
                    "is_misplaced": item_is_misplaced,

                    "bbox": item["bbox"],
                    "source_png": item["source_png"],
                    "angle_info": item["angle_info"],
                    "depth_row": item.get("depth_row", "front"),
                    "position_index": item.get("position_index"),
                })
                object_id += 1

    result.alpha_composite(ctx["combined_occluder"])
    return result, objects, slot_labels


# -----------------------------------------------------------------------------
# 6. YOLO 변환 / 확인 유틸
# -----------------------------------------------------------------------------

def objects_to_yolo_lines(
    background: Image.Image,
    objects: List[Dict[str, Any]],
    class_map: Dict[str, int],
    use_front_only: bool = True,
) -> List[str]:
    """objects를 YOLO txt 라인으로 변환만 한다. 파일 저장은 ipynb에서 한다."""
    image_w, image_h = background.size
    yolo_lines: List[str] = []

    for obj in objects:
        if use_front_only and obj.get("depth_row", "front") != "front":
            continue

        product_id = str(obj["product_id"])
        if product_id not in class_map:
            print("[class_map에 없는 product_id]", product_id)
            continue

        class_id = class_map[product_id]
        x1, y1, x2, y2 = obj["bbox"]

        x1 = max(0, min(x1, image_w))
        x2 = max(0, min(x2, image_w))
        y1 = max(0, min(y1, image_h))
        y2 = max(0, min(y2, image_h))

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            continue

        x_center = ((x1 + x2) / 2) / image_w
        y_center = ((y1 + y2) / 2) / image_h
        norm_w = box_w / image_w
        norm_h = box_h / image_h

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    return yolo_lines


def class_map_to_lines(class_map: Dict[str, int]) -> List[str]:
    """class_map을 txt 저장용 문자열 리스트로 변환한다. 파일 저장은 ipynb에서 한다."""
    lines = ["class_id\tproduct_id"]
    for product_id, class_id in sorted(class_map.items(), key=lambda item: item[1]):
        lines.append(f"{class_id}\t{product_id}")
    return lines


def show_image(image: Image.Image, title: str = "synthetic result", figsize: Tuple[int, int] = (16, 9)) -> None:
    """노트북에서 결과 이미지를 확인하는 헬퍼 함수."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()
