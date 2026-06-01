# -*- coding: utf-8 -*-
"""
synthetic_label_exporter.py

비었쇼 합성 데이터 공통 라벨 저장 모듈.
front / side 합성 공통 py에서 모두 호출할 수 있도록 만든 SAHI 기준 라벨 exporter입니다.

저장물 4종:
    1) 전체 선반 PNG 이미지
    2) 상품 탐지 YOLO txt        : full image 좌표 기준
    3) 선반 앞턱 YOLO-seg txt    : full image 좌표 기준
    4) slot_state JSON           : 후처리 / UI / 검증용 메타 라벨

권장 사용 예시:
    import synthetic_label_exporter as labeler

    labeler.save_sahi_labels(
        result=result,
        objects=objects,
        slot_labels=slot_labels,
        ctx=ctx,
        scenario_code=SCENARIO_CODE,
        scenario_name=SCENARIO_NAME,
        image_path=save_path,
        product_yolo_path=product_yolo_path,
        shelf_lip_yolo_path=shelf_lip_yolo_path,
        slot_json_path=json_path,
        seed=seed,
        view="side",   # 또는 "front"
    )

주의:
    - SAHI / slicing inference 기준이므로 crop 기준 라벨은 생성하지 않습니다.
    - 모든 bbox / polygon / slot geometry는 full image 좌표 기준으로 저장합니다.
    - crop 라벨이 필요해지면 이 파일을 확장해서 별도 파생 데이터로 만들면 됩니다.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import json
import math
import hashlib

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = Any  # type: ignore

Number = Union[int, float]
BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]

STATUS_LABEL_BY_CODE = {
    "NORMAL": "정상",
    "REPLENISH_REQUIRED": "보충 필요",
    "ORDER_REQUIRED": "발주 필요",
    "CHECK_REQUIRED": "확인 필요",
}

STATUS_CODE_BY_KO_KEYWORD = {
    "정상": "NORMAL",
    "보충": "REPLENISH_REQUIRED",
    "발주": "ORDER_REQUIRED",
    "확인": "CHECK_REQUIRED",
    "오진열": "CHECK_REQUIRED",
}

DEFAULT_STATUS_RULE = {
    "NORMAL": "no issue",
    "REPLENISH_REQUIRED": "missing exists and store stock is greater than reorder point",
    "ORDER_REQUIRED": "missing exists and store stock is less than or equal to reorder point",
    "CHECK_REQUIRED": "misplaced or abnormal display",
}

DEFAULT_SAHI_CONFIG = {
    "enabled": True,
    "slice_height": 640,
    "slice_width": 640,
    "overlap_height_ratio": 0.20,
    "overlap_width_ratio": 0.20,
    "postprocess_type": "NMS",
}

# ---------------------------------------------------------------------------
# 기본 유틸
# ---------------------------------------------------------------------------


def ensure_parent(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _to_builtin(value: Any) -> Any:
    """Path / numpy scalar / tuple 등을 JSON 저장 가능한 형태로 변환합니다."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(v) for v in value]
    # numpy scalar 대응
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    return value


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> Path:
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(data), f, ensure_ascii=False, indent=2)
    return path


def write_text_lines(lines: Sequence[str], path: Union[str, Path]) -> Path:
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def get_image_size(image: Any, ctx: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
    if hasattr(image, "size"):
        w, h = image.size
        return int(w), int(h)
    if ctx is not None and "background" in ctx and hasattr(ctx["background"], "size"):
        w, h = ctx["background"].size
        return int(w), int(h)
    raise ValueError("image 또는 ctx['background']에서 이미지 크기를 확인할 수 없습니다.")


def clamp(value: Number, low: Number, high: Number) -> float:
    return float(max(low, min(value, high)))


def clamp_bbox_xyxy(bbox: Sequence[Number], image_w: int, image_h: int) -> BBox:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = clamp(x1, 0, image_w)
    x2 = clamp(x2, 0, image_w)
    y1 = clamp(y1, 0, image_h)
    y2 = clamp(y2, 0, image_h)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def bbox_to_yolo(bbox: Sequence[Number], image_w: int, image_h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = clamp_bbox_xyxy(bbox, image_w, image_h)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return (
        cx / image_w if image_w else 0.0,
        cy / image_h if image_h else 0.0,
        bw / image_w if image_w else 0.0,
        bh / image_h if image_h else 0.0,
    )


def points_to_yolo(points: Sequence[Sequence[Number]], image_w: int, image_h: int) -> List[List[float]]:
    out: List[List[float]] = []
    for p in points:
        if len(p) < 2:
            continue
        x = clamp(float(p[0]), 0, image_w) / image_w if image_w else 0.0
        y = clamp(float(p[1]), 0, image_h) / image_h if image_h else 0.0
        out.append([x, y])
    return out


def flatten_polygon_yolo(points_yolo: Sequence[Sequence[Number]]) -> List[float]:
    flat: List[float] = []
    for x, y in points_yolo:
        flat.extend([float(x), float(y)])
    return flat


def get_bbox_xyxy(obj: Dict[str, Any]) -> Optional[BBox]:
    """front / side object 포맷을 모두 받아 bbox xyxy tuple로 변환합니다."""
    if obj.get("bbox_xyxy_full") is not None:
        b = obj["bbox_xyxy_full"]
        return tuple(float(v) for v in b[:4])  # type: ignore
    if obj.get("bbox_xyxy") is not None:
        b = obj["bbox_xyxy"]
        return tuple(float(v) for v in b[:4])  # type: ignore
    if obj.get("bbox") is not None:
        b = obj["bbox"]
        if isinstance(b, dict):
            return (
                float(b.get("x1", 0)),
                float(b.get("y1", 0)),
                float(b.get("x2", 0)),
                float(b.get("y2", 0)),
            )
        return tuple(float(v) for v in b[:4])  # type: ignore
    if obj.get("bbox_dict") is not None:
        b = obj["bbox_dict"]
        return (
            float(b.get("x1", 0)),
            float(b.get("y1", 0)),
            float(b.get("x2", 0)),
            float(b.get("y2", 0)),
        )
    return None


def get_depth(obj: Dict[str, Any]) -> str:
    return str(
        obj.get("depth_gt")
        or obj.get("depth_row")
        or obj.get("depth")
        or "front"
    )


def stable_int_from_str(text: str, min_value: int, max_value: int) -> int:
    """상품/슬롯별 가짜 재고값을 재현 가능하게 만들 때 사용합니다."""
    if max_value < min_value:
        min_value, max_value = max_value, min_value
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    n = int(h[:8], 16)
    return min_value + (n % (max_value - min_value + 1))


# ---------------------------------------------------------------------------
# ctx / slot / object 추출 유틸
# ---------------------------------------------------------------------------


def get_class_map(ctx: Optional[Dict[str, Any]] = None, class_map: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    if class_map is not None:
        return {str(k): int(v) for k, v in class_map.items()}
    if ctx is not None and ctx.get("class_map") is not None:
        return {str(k): int(v) for k, v in ctx["class_map"].items()}
    return {}


def get_all_slots_from_ctx(ctx: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ctx:
        return []
    for key in ["assigned_slots", "all_slots", "slots"]:
        if isinstance(ctx.get(key), list):
            return list(ctx[key])
    # front 구버전 all_zones 구조 대응
    slots: List[Dict[str, Any]] = []
    if isinstance(ctx.get("all_zones"), list):
        for zone in ctx["all_zones"]:
            for slot in zone.get("slots", []) or []:
                s = dict(slot)
                s.setdefault("category", zone.get("category"))
                s.setdefault("zone_id", zone.get("zone_id"))
                slots.append(s)
    return slots


def get_slot_map(ctx: Optional[Dict[str, Any]], slot_labels: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    slot_map: Dict[str, Dict[str, Any]] = {}
    for slot in get_all_slots_from_ctx(ctx):
        if slot.get("slot_id") is not None:
            slot_map[str(slot["slot_id"])] = dict(slot)
    if slot_labels:
        for label in slot_labels:
            sid = label.get("slot_id")
            if sid is None:
                continue
            base = slot_map.get(str(sid), {}).copy()
            base.update(label)
            slot_map[str(sid)] = base
    return slot_map


def get_shelf_lips_from_ctx(ctx: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    front / side ctx에서 shelf_lip 정보를 공통 포맷으로 가져옵니다.

    지원하는 key:
    - ctx["shelf_lips"]                : side 또는 표준 포맷
    - ctx["coords"]["shelf_lips"]      : side coords 포맷
    - ctx["shelf_lip_polygons"]        : front 기존 포맷
    """
    if not ctx:
        return []

    raw_lips = None

    # 1. 표준 key
    if isinstance(ctx.get("shelf_lips"), list):
        raw_lips = ctx["shelf_lips"]

    # 2. side coords key
    if raw_lips is None:
        coords = ctx.get("coords")
        if isinstance(coords, dict) and isinstance(coords.get("shelf_lips"), list):
            raw_lips = coords["shelf_lips"]

    # 3. front 기존 key
    if raw_lips is None and isinstance(ctx.get("shelf_lip_polygons"), list):
        raw_lips = ctx["shelf_lip_polygons"]

    if raw_lips is None:
        return []

    shelf_lips = []

    for idx, lip in enumerate(raw_lips, start=1):
        # dict 포맷
        if isinstance(lip, dict):
            points = (
                lip.get("points")
                or lip.get("points_xy")
                or lip.get("polygon")
                or lip.get("polygon_xy")
            )

            shelf_lip_id = (
                lip.get("shelf_lip_id")
                or lip.get("lip_id")
                or f"shelf_lip_{idx}"
            )

            shelf_no = lip.get("shelf_no") or lip.get("row_no") or idx

        # list 포맷: [[x1,y1], [x2,y2], ...]
        else:
            points = lip
            shelf_lip_id = f"shelf_lip_{idx}"
            shelf_no = idx

        if points is None:
            continue

        # points가 numpy array일 수도 있으니 list로 정리
        normalized_points = []
        for p in points:
            if p is None or len(p) < 2:
                continue
            normalized_points.append([float(p[0]), float(p[1])])

        if len(normalized_points) < 3:
            continue

        shelf_lips.append({
            "shelf_lip_id": shelf_lip_id,
            "lip_id": shelf_lip_id,
            "shelf_no": shelf_no,
            "points": normalized_points,
        })

    return shelf_lips


def get_background_name(ctx: Optional[Dict[str, Any]] = None, fallback: str = "") -> str:
    if ctx:
        paths = ctx.get("paths")
        if isinstance(paths, dict) and paths.get("background_path") is not None:
            return Path(paths["background_path"]).name
        if ctx.get("background_path") is not None:
            return Path(ctx["background_path"]).name
    return fallback


def get_slot_row_col(slot: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    row_no = slot.get("row_no")
    if row_no is None:
        row_hint = str(slot.get("row_hint", ""))
        if row_hint.startswith("row_"):
            try:
                row_no = int(row_hint.replace("row_", ""))
            except Exception:
                row_no = None
    col_no = slot.get("col_no")
    if col_no is None:
        col_no = slot.get("category_index")
    if col_no is None:
        col_no = slot.get("slot_index")
    try:
        row_no = int(row_no) if row_no is not None else None
    except Exception:
        row_no = None
    try:
        col_no = int(col_no) if col_no is not None else None
    except Exception:
        col_no = None
    return row_no, col_no


# ---------------------------------------------------------------------------
# geometry / front line / shelf lip
# ---------------------------------------------------------------------------


def _collect_slot_points(slot: Dict[str, Any]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []

    def add_point(p: Any):
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                points.append((float(p[0]), float(p[1])))
            except Exception:
                pass

    for key in ["anchor", "points"]:
        value = slot.get(key)
        if isinstance(value, list):
            # anchor는 [x,y], points는 [[x,y], ...] 모두 대응
            if len(value) >= 2 and all(isinstance(v, (int, float)) for v in value[:2]):
                add_point(value)
            else:
                for p in value:
                    add_point(p)

    for seg_key in ["slot_segment", "row_segment"]:
        seg = slot.get(seg_key)
        if isinstance(seg, dict):
            add_point(seg.get("start"))
            add_point(seg.get("end"))

    for key_pair in [("x1", "y1"), ("x2", "y2")]:
        if slot.get(key_pair[0]) is not None and slot.get(key_pair[1]) is not None:
            add_point([slot[key_pair[0]], slot[key_pair[1]]])

    # front 구버전 slot: x1,y1,x2,y2 rectangle
    if all(slot.get(k) is not None for k in ["x1", "y1", "x2", "y2"]):
        x1, y1, x2, y2 = [float(slot[k]) for k in ["x1", "y1", "x2", "y2"]]
        points.extend([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    return points


def build_slot_geometry(slot: Dict[str, Any], image_w: int, image_h: int, padding: int = 24) -> Dict[str, Any]:
    """slot bbox / polygon을 full image 좌표 기준으로 생성합니다."""
    # 명확한 rectangle이 있으면 우선 사용
    if all(slot.get(k) is not None for k in ["x1", "y1", "x2", "y2"]):
        x1, y1, x2, y2 = [float(slot[k]) for k in ["x1", "y1", "x2", "y2"]]
        bbox = clamp_bbox_xyxy([x1, y1, x2, y2], image_w, image_h)
        bx1, by1, bx2, by2 = bbox
        polygon = [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]]
    else:
        pts = _collect_slot_points(slot)
        if not pts:
            bbox = (0.0, 0.0, 0.0, 0.0)
            polygon = []
        else:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1 = min(xs) - padding
            y1 = min(ys) - padding
            x2 = max(xs) + padding
            y2 = max(ys) + padding
            # side slot_segment는 선 형태라 높이가 너무 얇아질 수 있어 최소 높이 보정
            if (y2 - y1) < 80:
                mid = (y1 + y2) / 2
                y1 = mid - 50
                y2 = mid + 90
            bbox = clamp_bbox_xyxy([x1, y1, x2, y2], image_w, image_h)
            bx1, by1, bx2, by2 = bbox
            polygon = [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]]

    bbox_yolo = bbox_to_yolo(bbox, image_w, image_h)
    return {
        "slot_bbox_xyxy": [int(round(v)) for v in bbox],
        "slot_polygon_xy": [[int(round(x)), int(round(y))] for x, y in polygon],
        "slot_bbox_yolo": [round(float(v), 6) for v in bbox_yolo],
    }


def build_front_lines(ctx: Optional[Dict[str, Any]], image_w: int, image_h: int, view: str = "side") -> List[Dict[str, Any]]:
    """front/back 판단용 row별 front line을 생성합니다."""
    lines: List[Dict[str, Any]] = []
    coords = ctx.get("coords") if ctx else None

    # side 기본 좌표: coords['slot_row_points'] 사용
    if isinstance(coords, dict) and isinstance(coords.get("slot_row_points"), list):
        for row in coords["slot_row_points"]:
            row_id = str(row.get("row_id", ""))
            points = row.get("points") or []
            if len(points) >= 2:
                try:
                    row_no = int(row_id.replace("row_", "")) if row_id.startswith("row_") else None
                except Exception:
                    row_no = None
                p1 = points[0]
                p2 = points[-1]
                lines.append({
                    "line_id": f"row_{row_no}_front_line" if row_no is not None else f"{row_id}_front_line",
                    "row_no": row_no,
                    "points_xy": [[int(p1[0]), int(p1[1])], [int(p2[0]), int(p2[1])]],
                    "source": "slot_row_points",
                    "front_band_px": 80,
                    "back_band_px": 160,
                })
        if lines:
            return lines

    # front / fallback: slot rectangle의 y2 기준으로 row별 horizontal line 생성
    slots = get_all_slots_from_ctx(ctx)
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for slot in slots:
        row_no, _ = get_slot_row_col(slot)
        if row_no is None:
            continue
        grouped.setdefault(row_no, []).append(slot)

    for row_no, row_slots in sorted(grouped.items()):
        xs: List[float] = []
        bottoms: List[float] = []
        for s in row_slots:
            if all(s.get(k) is not None for k in ["x1", "x2", "y2"]):
                xs.extend([float(s["x1"]), float(s["x2"])])
                bottoms.append(float(s["y2"]))
            else:
                pts = _collect_slot_points(s)
                if pts:
                    xs.extend([p[0] for p in pts])
                    bottoms.append(max(p[1] for p in pts))
        if xs and bottoms:
            x1 = clamp(min(xs), 0, image_w)
            x2 = clamp(max(xs), 0, image_w)
            y = clamp(sum(bottoms) / len(bottoms), 0, image_h)
            lines.append({
                "line_id": f"row_{row_no}_front_line",
                "row_no": row_no,
                "points_xy": [[int(round(x1)), int(round(y))], [int(round(x2)), int(round(y))]],
                "source": "slot_geometry",
                "front_band_px": 80,
                "back_band_px": 160,
            })

    return lines


def build_shelf_lips(ctx: Optional[Dict[str, Any]], image_w: int, image_h: int) -> List[Dict[str, Any]]:
    shelf_lips: List[Dict[str, Any]] = []
    for idx, lip in enumerate(get_shelf_lips_from_ctx(ctx), start=1):
        pts = lip.get("points") or lip.get("points_xy") or []
        if len(pts) < 3:
            continue
        points_xy = [[int(round(float(p[0]))), int(round(float(p[1])))] for p in pts]
        points_yolo = points_to_yolo(points_xy, image_w, image_h)
        shelf_lips.append({
            "shelf_lip_id": lip.get("lip_id") or lip.get("shelf_lip_id") or f"shelf_lip_{idx}",
            "shelf_no": lip.get("shelf_no") or idx,
            "points_xy": points_xy,
            "points_yolo": [[round(float(x), 6), round(float(y), 6)] for x, y in points_yolo],
            "included_in_shelf_lip_yolo_seg": True,
        })
    return shelf_lips


def build_shelf_lip_yolo_seg_lines(shelf_lips: Sequence[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for lip in shelf_lips:
        if not lip.get("included_in_shelf_lip_yolo_seg", True):
            continue
        points_yolo = lip.get("points_yolo") or []
        if len(points_yolo) < 3:
            continue
        flat = flatten_polygon_yolo(points_yolo)
        lines.append("0 " + " ".join(f"{v:.6f}" for v in flat))
    return lines


# ---------------------------------------------------------------------------
# object / slot 표준화
# ---------------------------------------------------------------------------


def standardize_objects(
    objects: Sequence[Dict[str, Any]],
    class_map: Dict[str, int],
    image_w: int,
    image_h: int,
    slot_label_map: Optional[Dict[str, Dict[str, Any]]] = None,
    include_back_for_product_yolo: bool = True,
) -> List[Dict[str, Any]]:
    slot_label_map = slot_label_map or {}
    out: List[Dict[str, Any]] = []

    for idx, obj in enumerate(objects, start=1):
        bbox = get_bbox_xyxy(obj)
        if bbox is None:
            continue
        bbox = clamp_bbox_xyxy(bbox, image_w, image_h)
        x1, y1, x2, y2 = bbox
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue

        slot_id = str(obj.get("slot_id", "")) if obj.get("slot_id") is not None else ""
        slot_label = slot_label_map.get(slot_id, {})

        product_id = str(obj.get("product_id") or obj.get("actual_product_id") or "")
        product_name = str(obj.get("product_name") or obj.get("actual_product_name") or "")
        class_id = obj.get("class_id")
        if class_id is None and product_id in class_map:
            class_id = class_map[product_id]

        depth = get_depth(obj)
        visible = bool(obj.get("visible", True))
        included = bool(obj.get("included_in_product_yolo", True))
        if not include_back_for_product_yolo and depth != "front":
            included = False
        if class_id is None:
            included = False

        bbox_yolo = bbox_to_yolo(bbox, image_w, image_h)
        center_xy = [int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))]
        bottom_center_xy = [int(round((x1 + x2) / 2)), int(round(y2))]

        expected_product_id = obj.get("expected_product_id") or obj.get("target_product_id") or slot_label.get("expected_product_id") or slot_label.get("target_product_id")
        expected_product_name = obj.get("expected_product_name") or obj.get("target_product_name") or slot_label.get("expected_product_name") or slot_label.get("target_product_name")
        actual_product_id = obj.get("actual_product_id") or slot_label.get("actual_product_id") or product_id
        actual_product_name = obj.get("actual_product_name") or slot_label.get("actual_product_name") or product_name

        out.append({
            "object_id": int(obj.get("object_id", idx)),
            "slot_id": slot_id,
            "product_id": product_id,
            "product_name": product_name,
            "class_id": int(class_id) if class_id is not None else None,

            "expected_product_id": str(expected_product_id) if expected_product_id is not None else None,
            "expected_product_name": str(expected_product_name) if expected_product_name is not None else None,
            "actual_product_id": str(actual_product_id) if actual_product_id is not None else None,
            "actual_product_name": str(actual_product_name) if actual_product_name is not None else None,

            "scenario_code": obj.get("scenario_code") or slot_label.get("scenario_code"),
            "sub_scenario_code": obj.get("sub_scenario_code") or slot_label.get("sub_scenario_code") or obj.get("scenario_code") or slot_label.get("scenario_code"),
            "action": obj.get("action") or slot_label.get("action"),

            "depth_gt": depth,
            "position_index": obj.get("position_index"),
            "lane_index": obj.get("lane_index"),
            "stack_mode": obj.get("stack_mode", "normal"),
            "stack_level": int(obj.get("stack_level", 0) or 0),

            "bbox_xyxy_full": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "bbox_yolo_full": [round(float(v), 6) for v in bbox_yolo],
            "bbox_center_xy": center_xy,
            "bbox_bottom_center_xy": bottom_center_xy,

            "visible": visible,
            "included_in_product_yolo": included,

            "is_misplaced": bool(obj.get("is_misplaced", slot_label.get("is_misplaced", False))),
            "is_misplaced_item": bool(obj.get("is_misplaced_item", obj.get("is_misplaced", False))),

            "source_png": str(obj.get("source_png") or obj.get("selected_png_path") or ""),
        })

    return out


def build_product_yolo_lines(std_objects: Sequence[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for obj in std_objects:
        if not obj.get("included_in_product_yolo", True):
            continue
        if not obj.get("visible", True):
            continue
        class_id = obj.get("class_id")
        bbox_yolo = obj.get("bbox_yolo_full")
        if class_id is None or not bbox_yolo:
            continue
        lines.append(
            f"{int(class_id)} " + " ".join(f"{float(v):.6f}" for v in bbox_yolo)
        )
    return lines


def infer_status_code(slot_item: Dict[str, Any], missing_qty: int, is_misplaced: bool, store_stock_qty: int, reorder_point: int) -> str:
    # 명시 status_code 우선
    if slot_item.get("status_code"):
        return str(slot_item["status_code"])

    # 기존 final_status 한글 문자열 매핑
    final_status = str(slot_item.get("final_status") or slot_item.get("status_label") or "")
    for keyword, code in STATUS_CODE_BY_KO_KEYWORD.items():
        if keyword in final_status:
            return code

    if is_misplaced:
        return "CHECK_REQUIRED"
    if missing_qty <= 0:
        return "NORMAL"
    if store_stock_qty <= reorder_point:
        return "ORDER_REQUIRED"
    return "REPLENISH_REQUIRED"


def build_standard_slots(
    slot_labels: Sequence[Dict[str, Any]],
    objects: Sequence[Dict[str, Any]],
    ctx: Optional[Dict[str, Any]],
    image_w: int,
    image_h: int,
    scenario_code: str,
    scenario_name: str,
    store_id: str,
    shelf_id: str,
    detected_at: str,
) -> List[Dict[str, Any]]:
    slot_map = get_slot_map(ctx, slot_labels)

    # slot별 object count 집계
    obj_by_slot: Dict[str, List[Dict[str, Any]]] = {}
    for obj in objects:
        sid = str(obj.get("slot_id", ""))
        obj_by_slot.setdefault(sid, []).append(obj)

    std_slots: List[Dict[str, Any]] = []

    for label in slot_labels:
        slot_id = str(label.get("slot_id"))
        merged_slot = slot_map.get(slot_id, {}).copy()
        merged_slot.update(label)

        row_no, col_no = get_slot_row_col(merged_slot)
        slot_objects = obj_by_slot.get(slot_id, [])
        front_count = sum(1 for obj in slot_objects if get_depth(obj) == "front")
        back_count = sum(1 for obj in slot_objects if get_depth(obj) == "back")

        required_front = int(
            merged_slot.get("required_front_qty")
            or merged_slot.get("normal_front_qty")
            or merged_slot.get("display_qty")
            or merged_slot.get("front_display_qty")
            or front_count
            or 0
        )
        required_back = int(
            merged_slot.get("required_back_qty")
            or merged_slot.get("normal_back_qty")
            or merged_slot.get("back_display_qty")
            or required_front
            or 0
        )

        front_display_qty = int(merged_slot.get("front_display_qty", front_count))
        back_display_qty = int(merged_slot.get("back_display_qty", back_count))

        front_missing_qty = int(
            merged_slot.get("front_missing_qty")
            if merged_slot.get("front_missing_qty") is not None
            else max(0, required_front - front_display_qty)
        )
        back_missing_qty = int(
            merged_slot.get("back_missing_qty")
            if merged_slot.get("back_missing_qty") is not None
            else max(0, required_back - back_display_qty)
        )
        missing_qty = int(
            merged_slot.get("missing_qty")
            if merged_slot.get("missing_qty") is not None
            else front_missing_qty + back_missing_qty
        )

        total_required = max(1, required_front + required_back)
        missing_ratio = round(missing_qty / total_required, 4)

        expected_product_id = merged_slot.get("expected_product_id") or merged_slot.get("target_product_id")
        expected_product_name = merged_slot.get("expected_product_name") or merged_slot.get("target_product_name")
        actual_product_id = merged_slot.get("actual_product_id")
        if actual_product_id is None:
            actual_product_id = expected_product_id if not merged_slot.get("is_slot_empty") else None
        actual_product_name = merged_slot.get("actual_product_name")
        if actual_product_name is None:
            actual_product_name = expected_product_name if actual_product_id is not None else None

        is_misplaced = bool(merged_slot.get("is_misplaced", False))
        is_slot_empty = bool(merged_slot.get("is_slot_empty", False)) or (front_display_qty == 0 and back_display_qty == 0 and required_front + required_back > 0)
        is_front_depleted = bool(merged_slot.get("is_front_depleted", False)) or (required_front > 0 and front_display_qty == 0)

        # 실제 POS가 없는 합성 데이터이므로 재현 가능한 demo stock 생성
        store_stock_qty = int(merged_slot.get("store_stock_qty", stable_int_from_str(slot_id + str(expected_product_id), 0, 30)))
        reorder_point = int(merged_slot.get("reorder_point", 5))
        status_code = infer_status_code(merged_slot, missing_qty, is_misplaced, store_stock_qty, reorder_point)
        status_label = STATUS_LABEL_BY_CODE.get(status_code, str(merged_slot.get("status_label", status_code)))
        list_up = bool(merged_slot.get("list_up", status_code != "NORMAL"))

        if status_code == "CHECK_REQUIRED":
            base_score = 90
        elif status_code == "ORDER_REQUIRED":
            base_score = 85
        elif status_code == "REPLENISH_REQUIRED":
            base_score = 65
        else:
            base_score = 0
        priority_score = int(merged_slot.get("priority_score", min(100, base_score + int(missing_ratio * 30))))
        if priority_score >= 80:
            priority_label = "high"
        elif priority_score >= 50:
            priority_label = "medium"
        elif priority_score > 0:
            priority_label = "low"
        else:
            priority_label = "none"

        recommended_replenish_qty = int(merged_slot.get("recommended_replenish_qty", max(0, front_missing_qty)))
        recommended_order_qty = int(merged_slot.get("recommended_order_qty", 0 if store_stock_qty > reorder_point else max(0, missing_qty - store_stock_qty)))

        std_slots.append({
            "slot_id": slot_id,
            "shelf_id": shelf_id,
            "shelf_no": int(merged_slot.get("shelf_no", 1) or 1),
            "row_no": row_no,
            "col_no": col_no,
            "location_label": merged_slot.get("location_label") or f"선반 A {row_no}-{col_no}",

            "slot_geometry": build_slot_geometry(merged_slot, image_w, image_h),

            "scenario_code": scenario_code,
            "sub_scenario_code": merged_slot.get("sub_scenario_code") or merged_slot.get("scenario_code") or scenario_code,
            "action": merged_slot.get("action", "normal"),

            "expected_product_id": str(expected_product_id) if expected_product_id is not None else None,
            "expected_product_name": str(expected_product_name) if expected_product_name is not None else None,
            "actual_product_id": str(actual_product_id) if actual_product_id is not None else None,
            "actual_product_name": str(actual_product_name) if actual_product_name is not None else None,

            "required_front_qty": required_front,
            "required_back_qty": required_back,
            "front_display_qty": front_display_qty,
            "back_display_qty": back_display_qty,
            "front_missing_qty": front_missing_qty,
            "back_missing_qty": back_missing_qty,
            "missing_qty": missing_qty,
            "missing_ratio": missing_ratio,
            "front_missing_indices": merged_slot.get("front_missing_indices", []),
            "back_missing_indices": merged_slot.get("back_missing_indices", []),
            "target_column_index": merged_slot.get("target_column_index"),

            "is_misplaced": is_misplaced,
            "is_front_depleted": is_front_depleted,
            "is_slot_empty": is_slot_empty,

            "store_stock_qty": store_stock_qty,
            "reorder_point": reorder_point,
            "recommended_replenish_qty": recommended_replenish_qty,
            "recommended_order_qty": recommended_order_qty,

            "status_code": status_code,
            "status_label": status_label,
            "list_up": list_up,
            "priority_score": priority_score,
            "priority_label": priority_label,
            "detected_at": detected_at,
        })

    return std_slots


def build_planogram(slots: Sequence[Dict[str, Any]], planogram_id: str, shelf_id: str, view: str) -> Dict[str, Any]:
    return {
        "planogram_id": planogram_id,
        "shelf_id": shelf_id,
        "view": view,
        "slots": [
            {
                "slot_id": s.get("slot_id"),
                "row_no": s.get("row_no"),
                "col_no": s.get("col_no"),
                "expected_product_id": s.get("expected_product_id"),
                "expected_product_name": s.get("expected_product_name"),
                "expected_front_count": s.get("required_front_qty"),
                "expected_back_count": s.get("required_back_qty"),
            }
            for s in slots
        ],
    }


def build_overview(slots: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    def count_status(code: str) -> int:
        return sum(1 for s in slots if s.get("status_code") == code)

    return {
        "total_slot_count": len(slots),
        "normal_slot_count": count_status("NORMAL"),
        "replenish_required_count": count_status("REPLENISH_REQUIRED"),
        "order_required_count": count_status("ORDER_REQUIRED"),
        "check_required_count": count_status("CHECK_REQUIRED"),
        "list_up_count": sum(1 for s in slots if s.get("list_up")),
        "misplaced_slot_count": sum(1 for s in slots if s.get("is_misplaced")),
        "empty_slot_count": sum(1 for s in slots if s.get("is_slot_empty")),
        "front_depleted_slot_count": sum(1 for s in slots if s.get("is_front_depleted")),
    }


def build_work_list(slots: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    targets = [s for s in slots if s.get("list_up")]
    targets = sorted(targets, key=lambda s: int(s.get("priority_score", 0)), reverse=True)
    out: List[Dict[str, Any]] = []
    for rank, s in enumerate(targets, start=1):
        out.append({
            "rank": rank,
            "slot_id": s.get("slot_id"),
            "product_id": s.get("expected_product_id") or s.get("actual_product_id"),
            "product_name": s.get("expected_product_name") or s.get("actual_product_name"),
            "status_code": s.get("status_code"),
            "status_label": s.get("status_label"),
            "store_stock_qty": s.get("store_stock_qty"),
            "recommended_replenish_qty": s.get("recommended_replenish_qty"),
            "recommended_order_qty": s.get("recommended_order_qty"),
            "location_label": s.get("location_label"),
            "detected_at": s.get("detected_at"),
            "priority_score": s.get("priority_score"),
            "priority_label": s.get("priority_label"),
        })
    return out


def build_sku_detail_candidates(slots: Sequence[Dict[str, Any]], objects: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # slot별 대표 object source_png 가져오기
    source_by_slot: Dict[str, str] = {}
    for obj in objects:
        sid = str(obj.get("slot_id", ""))
        if sid and sid not in source_by_slot and obj.get("source_png"):
            source_by_slot[sid] = str(obj.get("source_png"))

    candidates: List[Dict[str, Any]] = []
    for s in slots:
        if not s.get("list_up"):
            continue
        sid = str(s.get("slot_id"))
        sku_id = s.get("expected_product_id") or s.get("actual_product_id")
        sku_name = s.get("expected_product_name") or s.get("actual_product_name")
        candidates.append({
            "slot_id": sid,
            "sku_id": sku_id,
            "sku_name": sku_name,
            "product_image_path": source_by_slot.get(sid, ""),
            "current_status": s.get("status_label"),
            "detected_at": s.get("detected_at"),
            "store_stock_qty": s.get("store_stock_qty"),
            "recommended_replenish_qty": s.get("recommended_replenish_qty"),
            "recommended_order_qty": s.get("recommended_order_qty"),
            "location_label": s.get("location_label"),
            "confidence": {
                "source": "synthetic_demo",
                "value": 0.65,
            },
        })
    return candidates


# ---------------------------------------------------------------------------
# 메인 저장 함수
# ---------------------------------------------------------------------------


def build_slot_state_json(
    *,
    image_name: str,
    image_path: Union[str, Path],
    image_w: int,
    image_h: int,
    objects: Sequence[Dict[str, Any]],
    slot_labels: Sequence[Dict[str, Any]],
    ctx: Optional[Dict[str, Any]],
    scenario_code: str,
    scenario_name: str,
    seed: Optional[int],
    view: str,
    class_map: Optional[Dict[str, int]] = None,
    store_id: str = "STORE_001",
    camera_id: str = "CAM_001",
    shelf_id: str = "SHELF_A",
    captured_at: Optional[str] = None,
    background_name: Optional[str] = None,
    planogram_id: Optional[str] = None,
    include_back_for_product_yolo: bool = True,
    sahi_config: Optional[Dict[str, Any]] = None,
    settings_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    detected_at = captured_at or datetime.now().strftime("%Y-%m-%d %H:%M")
    class_map_ = get_class_map(ctx, class_map)
    slot_map = get_slot_map(ctx, slot_labels)

    std_slots = build_standard_slots(
        slot_labels=slot_labels,
        objects=objects,
        ctx=ctx,
        image_w=image_w,
        image_h=image_h,
        scenario_code=scenario_code,
        scenario_name=scenario_name,
        store_id=store_id,
        shelf_id=shelf_id,
        detected_at=detected_at,
    )
    slot_label_map = {str(s.get("slot_id")): s for s in std_slots}

    std_objects = standardize_objects(
        objects=objects,
        class_map=class_map_,
        image_w=image_w,
        image_h=image_h,
        slot_label_map=slot_label_map,
        include_back_for_product_yolo=include_back_for_product_yolo,
    )

    shelf_lips = build_shelf_lips(ctx, image_w, image_h)
    front_lines = build_front_lines(ctx, image_w, image_h, view=view)
    planogram_id = planogram_id or f"{view.upper()}_{shelf_id}_V1"

    settings = {
        "generator_version": f"{view}_synthetic_v1",
        "modeling_strategy": "sahi_slicing_inference",
        "coordinate_system": "full_image",
        "product_yolo_include_rule": "visible front and back products are included" if include_back_for_product_yolo else "visible front products are included",
        "shelf_lip_class": {"0": "shelf_lip"},
        "sahi_config": sahi_config or DEFAULT_SAHI_CONFIG,
        "status_rule": DEFAULT_STATUS_RULE,
    }
    if settings_extra:
        settings.update(settings_extra)

    return {
        "image": {
            "image_id": Path(image_name).stem,
            "file_name": image_name,
            "image_path": str(image_path),
            "view": view,
            "camera_id": camera_id,
            "store_id": store_id,
            "shelf_id": shelf_id,
            "scenario_code": scenario_code,
            "scenario_name": scenario_name,
            "created_seed": seed,
            "captured_at": detected_at,
            "image_width": image_w,
            "image_height": image_h,
            "background_name": background_name or get_background_name(ctx),
        },
        "overview": build_overview(std_slots),
        "planogram": build_planogram(std_slots, planogram_id=planogram_id, shelf_id=shelf_id, view=view),
        "front_lines": front_lines,
        "slots": std_slots,
        "objects": std_objects,
        "shelf_lips": shelf_lips,
        "work_list": build_work_list(std_slots),
        "sku_detail_candidates": build_sku_detail_candidates(std_slots, std_objects),
        "class_map": class_map_,
        "settings": settings,
    }


def save_sahi_labels(
    *,
    result: Any,
    objects: Sequence[Dict[str, Any]],
    slot_labels: Sequence[Dict[str, Any]],
    ctx: Optional[Dict[str, Any]],
    scenario_code: str,
    scenario_name: str,
    image_path: Union[str, Path],
    product_yolo_path: Union[str, Path],
    shelf_lip_yolo_path: Union[str, Path],
    slot_json_path: Union[str, Path],
    seed: Optional[int] = None,
    view: str = "side",
    class_map: Optional[Dict[str, int]] = None,
    store_id: str = "STORE_001",
    camera_id: str = "CAM_001",
    shelf_id: str = "SHELF_A",
    captured_at: Optional[str] = None,
    background_name: Optional[str] = None,
    planogram_id: Optional[str] = None,
    include_back_for_product_yolo: bool = True,
    sahi_config: Optional[Dict[str, Any]] = None,
    settings_extra: Optional[Dict[str, Any]] = None,
    save_image: bool = True,
) -> Dict[str, Any]:
    """
    SAHI 기준 최종 라벨 3종 + 이미지 저장.

    Parameters
    ----------
    result:
        합성 결과 PIL Image.
    objects:
        render_from_slot_plan 결과 objects.
    slot_labels:
        render_from_slot_plan 결과 slot_labels.
    ctx:
        create_synthetic_context 결과. ctx가 없어도 일부 기능은 동작하지만,
        slot_geometry / shelf_lip / front_lines 생성을 위해 전달하는 것을 권장.
    image_path:
        저장할 전체 이미지 경로.
    product_yolo_path:
        상품 YOLO txt 저장 경로.
    shelf_lip_yolo_path:
        shelf_lip YOLO-seg txt 저장 경로.
    slot_json_path:
        slot_state JSON 저장 경로.
    include_back_for_product_yolo:
        SAHI 기준은 True 권장. 앞/뒤 보이는 상품을 모두 학습 라벨에 포함합니다.
    """
    image_path = Path(image_path)
    product_yolo_path = Path(product_yolo_path)
    shelf_lip_yolo_path = Path(shelf_lip_yolo_path)
    slot_json_path = Path(slot_json_path)

    image_w, image_h = get_image_size(result, ctx)
    image_name = image_path.name

    if save_image:
        ensure_parent(image_path)
        # PIL Image면 RGB 저장, 아니면 저장 생략
        if hasattr(result, "convert"):
            result.convert("RGB").save(image_path)

    label_json = build_slot_state_json(
        image_name=image_name,
        image_path=image_path,
        image_w=image_w,
        image_h=image_h,
        objects=objects,
        slot_labels=slot_labels,
        ctx=ctx,
        scenario_code=scenario_code,
        scenario_name=scenario_name,
        seed=seed,
        view=view,
        class_map=class_map,
        store_id=store_id,
        camera_id=camera_id,
        shelf_id=shelf_id,
        captured_at=captured_at,
        background_name=background_name,
        planogram_id=planogram_id,
        include_back_for_product_yolo=include_back_for_product_yolo,
        sahi_config=sahi_config,
        settings_extra=settings_extra,
    )

    product_lines = build_product_yolo_lines(label_json["objects"])
    shelf_lip_lines = build_shelf_lip_yolo_seg_lines(label_json["shelf_lips"])

    write_text_lines(product_lines, product_yolo_path)
    write_text_lines(shelf_lip_lines, shelf_lip_yolo_path)
    save_json(label_json, slot_json_path)

    return {
        "image_path": str(image_path),
        "product_yolo_path": str(product_yolo_path),
        "shelf_lip_yolo_path": str(shelf_lip_yolo_path),
        "slot_json_path": str(slot_json_path),
        "n_product_yolo_objects": len(product_lines),
        "n_shelf_lips": len(shelf_lip_lines),
        "n_slots": len(label_json["slots"]),
        "n_objects": len(label_json["objects"]),
        "label_json": label_json,
    }


# 기존 시나리오 노트북에서 이름을 더 직관적으로 쓰고 싶을 때 사용하는 alias
save_standard_outputs = save_sahi_labels


# ---------------------------------------------------------------------------
# metadata 저장 보조 함수
# ---------------------------------------------------------------------------


def save_product_class_map_tsv(class_map: Dict[str, int], path: Union[str, Path], product_name_map: Optional[Dict[str, str]] = None) -> Path:
    product_name_map = product_name_map or {}
    lines = ["class_id\tproduct_id\tproduct_name"]
    for product_id, class_id in sorted(class_map.items(), key=lambda x: int(x[1])):
        lines.append(f"{int(class_id)}\t{product_id}\t{product_name_map.get(str(product_id), '')}")
    return write_text_lines(lines, path)


__all__ = [
    "save_sahi_labels",
    "save_standard_outputs",
    "build_slot_state_json",
    "build_product_yolo_lines",
    "build_shelf_lip_yolo_seg_lines",
    "build_slot_geometry",
    "build_front_lines",
    "build_shelf_lips",
    "save_product_class_map_tsv",
]
