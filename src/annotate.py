from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from annotation_io import (
    CLASS_COLORS_BGR,
    CLASS_NAMES,
    load_label_mask,
    load_manifest,
    read_metadata,
    save_annotation,
    update_manifest_record,
)
from common import load_yaml, project_input_path, project_path, read_image, to_bgr_uint8

WINDOW_NAME = "Candy annotation"
TOP_BAR = 76
BOTTOM_BAR = 36


@dataclass
class Settings:
    project_config: Path
    ui_config: Path
    mode: str
    input_root: Path
    annotation_root: Path
    manifest_path: Path
    window_width: int
    window_height: int
    initial_brush_radius: int
    overlay_alpha: float
    autosave_on_navigation: bool
    max_undo_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ручная разметка пикселей для сегментатора.")
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/stage2.yaml"))
    parser.add_argument("--mode", choices=["sample", "raw"], default="sample")
    parser.add_argument("--input", type=Path, help="Явный путь к изображениям")
    parser.add_argument("--start", type=str, help="Начать с пути или части имени")
    parser.add_argument("--check", action="store_true", help="Только проверить готовность набора")
    return parser.parse_args()


def load_settings(args: argparse.Namespace) -> Settings:
    project = load_yaml(args.project_config)
    ui = load_yaml(args.config)
    input_root = (args.input or project_input_path(project, args.mode)).resolve()
    annotation_root = project_path(project, "annotations", "annotations").resolve()
    return Settings(
        project_config=args.project_config.resolve(),
        ui_config=args.config.resolve(),
        mode=args.mode,
        input_root=input_root,
        annotation_root=annotation_root,
        manifest_path=annotation_root / "manifest.csv",
        window_width=int(ui.get("window_width", 1400)),
        window_height=int(ui.get("window_height", 850)),
        initial_brush_radius=int(ui.get("initial_brush_radius", 18)),
        overlay_alpha=float(ui.get("overlay_alpha", 0.45)),
        autosave_on_navigation=bool(ui.get("autosave_on_navigation", True)),
        max_undo_steps=max(1, int(ui.get("max_undo_steps", 12))),
    )


def available_records(settings: Settings) -> tuple[list[dict[str, str]], list[str]]:
    records = load_manifest(settings.manifest_path)
    available: list[dict[str, str]] = []
    missing: list[str] = []
    for record in records:
        relative = record.get("source_relative_path", "")
        if not relative:
            continue
        if (settings.input_root / relative).exists():
            available.append(record)
        else:
            missing.append(relative)
    return available, missing


def check_dataset(settings: Settings) -> int:
    print(f"Режим: {settings.mode}")
    print(f"Источник: {settings.input_root}")
    print(f"Папка разметки: {settings.annotation_root}")
    print(f"Манифест: {settings.manifest_path}")
    if not settings.input_root.exists():
        print("ОШИБКА: папка источника не найдена.", file=sys.stderr)
        return 2
    if not settings.manifest_path.exists():
        print("ОШИБКА: manifest.csv не найден. Сначала выполни prepare.", file=sys.stderr)
        return 3
    records, missing = available_records(settings)
    print(f"Доступных изображений из манифеста: {len(records)}")
    print(f"Недоступных в текущем режиме: {len(missing)}")
    if missing:
        for item in missing[:10]:
            print(f"- недоступно: {item}")
        if len(missing) > 10:
            print(f"... ещё {len(missing) - 10}")
    return 0 if records else 4


class Annotator:
    def __init__(self, settings: Settings, start: str | None) -> None:
        self.settings = settings
        self.records, self.missing = available_records(settings)
        if not self.records:
            raise RuntimeError("В манифесте нет изображений, доступных в выбранном режиме.")

        self.index = 0
        if start:
            lowered = start.lower()
            for index, record in enumerate(self.records):
                if lowered in record["source_relative_path"].lower():
                    self.index = index
                    break

        self.current_class = 2
        self.brush_radius = settings.initial_brush_radius
        self.overlay_alpha = settings.overlay_alpha
        self.show_overlay = True
        self.show_help = False
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.drag_mode: str | None = None
        self.last_image_point: tuple[int, int] | None = None
        self.pan_previous: tuple[int, int] | None = None
        self.last_mouse_canvas = (0, 0)
        self.undo_stack: list[np.ndarray] = []
        self.dirty = False
        self.status = "not_started"
        self.relative_path = Path()
        self.image_path = Path()
        self.image = np.empty((0, 0, 3), dtype=np.uint8)
        self.mask = np.empty((0, 0), dtype=np.uint8)
        self.load_current()

    @property
    def view_width(self) -> int:
        return self.settings.window_width

    @property
    def view_height(self) -> int:
        return self.settings.window_height - TOP_BAR - BOTTOM_BAR

    def load_current(self) -> None:
        record = self.records[self.index]
        self.relative_path = Path(record["source_relative_path"])
        self.image_path = self.settings.input_root / self.relative_path
        self.image = to_bgr_uint8(read_image(self.image_path))
        self.mask = load_label_mask(
            self.settings.annotation_root,
            self.relative_path,
            self.image.shape[:2],
        )
        metadata = read_metadata(self.settings.annotation_root, self.relative_path)
        self.status = str(metadata.get("status", record.get("label_status", "not_started")))
        self.undo_stack.clear()
        self.dirty = False
        self.drag_mode = None
        self.last_image_point = None
        self.fit_view()
        print(
            f"Открыто [{self.index + 1}/{len(self.records)}]: "
            f"{self.relative_path.as_posix()} | status={self.status}"
        )

    def save(self) -> None:
        save_annotation(
            annotation_root=self.settings.annotation_root,
            relative_path=self.relative_path,
            mask=self.mask,
            status=self.status,
            source_mode=self.settings.mode,
            source_root=self.settings.input_root,
        )
        update_manifest_record(
            self.settings.manifest_path,
            self.relative_path,
            self.status,
            self.mask,
        )
        self.dirty = False
        print(f"Сохранено: {self.relative_path.as_posix()} | status={self.status}")

    def fit_view(self) -> None:
        image_height, image_width = self.image.shape[:2]
        self.scale = min(
            self.view_width / max(image_width, 1),
            self.view_height / max(image_height, 1),
        )
        self.scale = max(self.scale, 0.01)
        displayed_width = image_width * self.scale
        displayed_height = image_height * self.scale
        self.offset_x = (self.view_width - displayed_width) / 2.0
        self.offset_y = (self.view_height - displayed_height) / 2.0

    def clamp_view(self) -> None:
        image_height, image_width = self.image.shape[:2]
        displayed_width = image_width * self.scale
        displayed_height = image_height * self.scale

        if displayed_width <= self.view_width:
            self.offset_x = (self.view_width - displayed_width) / 2.0
        else:
            self.offset_x = min(0.0, max(self.view_width - displayed_width, self.offset_x))

        if displayed_height <= self.view_height:
            self.offset_y = (self.view_height - displayed_height) / 2.0
        else:
            self.offset_y = min(0.0, max(self.view_height - displayed_height, self.offset_y))

    def canvas_to_image(self, canvas_x: int, canvas_y: int) -> tuple[int, int] | None:
        local_y = canvas_y - TOP_BAR
        image_x = int((canvas_x - self.offset_x) / self.scale)
        image_y = int((local_y - self.offset_y) / self.scale)
        height, width = self.mask.shape
        if 0 <= image_x < width and 0 <= image_y < height:
            return image_x, image_y
        return None

    def push_undo(self) -> None:
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > self.settings.max_undo_steps:
            self.undo_stack.pop(0)

    def begin_stroke(self, point: tuple[int, int], erase: bool) -> None:
        self.push_undo()
        self.drag_mode = "erase" if erase else "paint"
        self.last_image_point = point
        self.apply_brush(point, point, erase=erase)

    def apply_brush(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        erase: bool,
    ) -> None:
        value = 0 if erase else self.current_class
        cv2.line(
            self.mask,
            start,
            end,
            int(value),
            thickness=max(1, self.brush_radius * 2),
            lineType=cv2.LINE_8,
        )
        cv2.circle(self.mask, end, self.brush_radius, int(value), -1, cv2.LINE_8)
        self.dirty = True
        if self.status == "not_started":
            self.status = "in_progress"

    def finish_stroke(self) -> None:
        self.drag_mode = None
        self.last_image_point = None

    def undo(self) -> None:
        if not self.undo_stack:
            print("Отменять нечего.")
            return
        self.mask = self.undo_stack.pop()
        self.dirty = True

    def navigate(self, step: int) -> None:
        if self.settings.autosave_on_navigation and (self.dirty or np.any(self.mask)):
            self.save()
        self.index = (self.index + step) % len(self.records)
        self.load_current()

    def next_incomplete(self) -> None:
        if self.settings.autosave_on_navigation and (self.dirty or np.any(self.mask)):
            self.save()
        start = self.index
        for offset in range(1, len(self.records) + 1):
            candidate = (start + offset) % len(self.records)
            record = self.records[candidate]
            metadata = read_metadata(
                self.settings.annotation_root,
                record["source_relative_path"],
            )
            status = str(metadata.get("status", record.get("label_status", "not_started")))
            if status != "complete":
                self.index = candidate
                self.load_current()
                return
        print("Все доступные изображения отмечены как complete.")

    def zoom_at(self, canvas_x: int, canvas_y: int, factor: float) -> None:
        point = self.canvas_to_image(canvas_x, canvas_y)
        old_scale = self.scale
        self.scale = min(8.0, max(0.01, self.scale * factor))
        if point is not None:
            image_x, image_y = point
            self.offset_x = canvas_x - image_x * self.scale
            self.offset_y = (canvas_y - TOP_BAR) - image_y * self.scale
        else:
            center_x = self.view_width / 2.0
            center_y = self.view_height / 2.0
            image_x = (center_x - self.offset_x) / old_scale
            image_y = (center_y - self.offset_y) / old_scale
            self.offset_x = center_x - image_x * self.scale
            self.offset_y = center_y - image_y * self.scale
        self.clamp_view()

    def on_mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        self.last_mouse_canvas = (x, y)
        ctrl_left = event == cv2.EVENT_LBUTTONDOWN and bool(flags & cv2.EVENT_FLAG_CTRLKEY)

        if event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16) & 0xFFFF
            if delta >= 0x8000:
                delta -= 0x10000
            self.zoom_at(x, y, 1.22 if delta > 0 else 1.0 / 1.22)
            return

        if event == cv2.EVENT_MBUTTONDOWN or ctrl_left:
            self.drag_mode = "pan"
            self.pan_previous = (x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.drag_mode == "pan":
            previous_x, previous_y = self.pan_previous or (x, y)
            self.offset_x += x - previous_x
            self.offset_y += y - previous_y
            self.pan_previous = (x, y)
            self.clamp_view()
            return

        if event in (cv2.EVENT_MBUTTONUP, cv2.EVENT_LBUTTONUP) and self.drag_mode == "pan":
            self.drag_mode = None
            self.pan_previous = None
            return

        point = self.canvas_to_image(x, y)
        if event == cv2.EVENT_LBUTTONDOWN and not ctrl_left and point is not None:
            self.begin_stroke(point, erase=False)
            return
        if event == cv2.EVENT_RBUTTONDOWN and point is not None:
            self.begin_stroke(point, erase=True)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.drag_mode in {"paint", "erase"} and point is not None:
            if self.last_image_point is not None:
                self.apply_brush(
                    self.last_image_point,
                    point,
                    erase=self.drag_mode == "erase",
                )
            self.last_image_point = point
            return

        if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP) and self.drag_mode in {"paint", "erase"}:
            self.finish_stroke()

    def visible_crop(self) -> tuple[np.ndarray, np.ndarray, int, int, int, int] | None:
        image_height, image_width = self.image.shape[:2]
        source_x0 = max(0, int(np.floor((-self.offset_x) / self.scale)))
        source_y0 = max(0, int(np.floor((-self.offset_y) / self.scale)))
        source_x1 = min(image_width, int(np.ceil((self.view_width - self.offset_x) / self.scale)))
        source_y1 = min(image_height, int(np.ceil((self.view_height - self.offset_y) / self.scale)))
        if source_x0 >= source_x1 or source_y0 >= source_y1:
            return None

        destination_x0 = max(0, int(round(self.offset_x + source_x0 * self.scale)))
        destination_y0 = max(0, int(round(self.offset_y + source_y0 * self.scale)))
        destination_x1 = min(self.view_width, int(round(self.offset_x + source_x1 * self.scale)))
        destination_y1 = min(self.view_height, int(round(self.offset_y + source_y1 * self.scale)))
        if destination_x0 >= destination_x1 or destination_y0 >= destination_y1:
            return None

        crop_image = self.image[source_y0:source_y1, source_x0:source_x1]
        crop_mask = self.mask[source_y0:source_y1, source_x0:source_x1]
        return (
            crop_image,
            crop_mask,
            destination_x0,
            destination_y0,
            destination_x1,
            destination_y1,
        )

    def render(self) -> np.ndarray:
        width = self.settings.window_width
        height = self.settings.window_height
        canvas = np.full((height, width, 3), 30, dtype=np.uint8)

        visible = self.visible_crop()
        if visible is not None:
            crop_image, crop_mask, x0, y0, x1, y1 = visible
            target_size = (x1 - x0, y1 - y0)
            interpolation = cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_NEAREST
            resized_image = cv2.resize(crop_image, target_size, interpolation=interpolation)
            if self.show_overlay:
                resized_mask = cv2.resize(crop_mask, target_size, interpolation=cv2.INTER_NEAREST)
                overlay = resized_image.copy()
                for class_id, color in CLASS_COLORS_BGR.items():
                    overlay[resized_mask == class_id] = color
                labeled = resized_mask > 0
                blended = cv2.addWeighted(
                    resized_image,
                    1.0 - self.overlay_alpha,
                    overlay,
                    self.overlay_alpha,
                    0.0,
                )
                resized_image[labeled] = blended[labeled]
            canvas[TOP_BAR + y0 : TOP_BAR + y1, x0:x1] = resized_image

        dirty_text = "*" if self.dirty else ""
        relative_text = self.relative_path.as_posix()
        title = (
            f"[{self.index + 1}/{len(self.records)}] {relative_text}{dirty_text} | "
            f"class={self.current_class}:{CLASS_NAMES[self.current_class]} | "
            f"brush={self.brush_radius}px | zoom={self.scale:.2f} | status={self.status}"
        )
        cv2.putText(canvas, title[:190], (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1, cv2.LINE_AA)

        legend_x = 12
        for class_id in (1, 2, 3):
            color = CLASS_COLORS_BGR[class_id]
            cv2.rectangle(canvas, (legend_x, 39), (legend_x + 22, 61), color, -1)
            cv2.putText(
                canvas,
                f"{class_id} {CLASS_NAMES[class_id]}",
                (legend_x + 30, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            legend_x += 210

        controls = (
            "1/2/3 class | left paint | right erase | wheel zoom | Ctrl+left/middle pan | "
            "+/- brush | S save | C complete | N/P | J incomplete | U undo | F fit | H help | Q quit"
        )
        cv2.putText(canvas, controls[:205], (10, height - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        point = self.canvas_to_image(*self.last_mouse_canvas)
        if point is not None and self.drag_mode != "pan":
            cx = int(round(self.offset_x + point[0] * self.scale))
            cy = TOP_BAR + int(round(self.offset_y + point[1] * self.scale))
            radius = max(2, int(round(self.brush_radius * self.scale)))
            cv2.circle(canvas, (cx, cy), radius, (255, 255, 255), 1, cv2.LINE_AA)

        if self.show_help:
            self.draw_help(canvas)
        return canvas

    def draw_help(self, canvas: np.ndarray) -> None:
        x0, y0 = 70, 100
        x1, y1 = self.settings.window_width - 70, self.settings.window_height - 70
        layer = canvas.copy()
        cv2.rectangle(layer, (x0, y0), (x1, y1), (12, 12, 12), -1)
        cv2.addWeighted(layer, 0.94, canvas, 0.06, 0.0, canvas)
        lines = [
            "PIXEL ANNOTATION",
            "",
            "1 background: belt, shadows, dirt, scratches and every non-candy pixel.",
            "2 candy_core: confident pixels inside the thick central body of a candy.",
            "3 sticker: confident pixels inside the white calibration sticker.",
            "",
            "Do not paint the whole image. Add varied, confident strokes; avoid uncertain boundaries.",
            "Wrapped and unwrapped candies are both candy_core at this stage.",
            "For border candies paint the visible central part. Do not paint long wrapper tails as core.",
            "",
            "Left paint | Right erase | Wheel zoom | Ctrl+Left or Middle pan",
            "+/- brush | U undo | V overlay | F fit | S save | C complete",
            "N/P next/previous | J next incomplete | I in_progress | Q/Esc save and quit",
            "",
            "Press H to close help.",
        ]
        y = y0 + 34
        for line in lines:
            cv2.putText(canvas, line, (x0 + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)
            y += 28

    def run(self) -> int:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.settings.window_width, self.settings.window_height)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        print("Программа разметки запущена. Нажми H для справки.")
        while True:
            cv2.imshow(WINDOW_NAME, self.render())
            key = cv2.waitKeyEx(20)
            if key == -1:
                try:
                    visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                except cv2.error:
                    # OpenCV 5 + Qt can briefly report a NULL guiReceiver
                    # immediately after the window is created. This is not a
                    # fatal error, so keep the editor loop alive.
                    visible = 1.0
                if visible < 1:
                    break
                continue

            low = key & 0xFF
            if low in (ord("q"), 27):
                if self.dirty or np.any(self.mask):
                    self.save()
                break
            if low in (ord("1"), ord("2"), ord("3")):
                self.current_class = int(chr(low))
            elif low in (ord("+"), ord("=")):
                self.brush_radius = min(300, self.brush_radius + max(1, self.brush_radius // 5))
            elif low in (ord("-"), ord("_")):
                self.brush_radius = max(1, self.brush_radius - max(1, self.brush_radius // 5))
            elif low == ord("s"):
                self.save()
            elif low == ord("c"):
                self.status = "complete"
                self.save()
            elif low == ord("i"):
                self.status = "in_progress"
                self.dirty = True
            elif low == ord("n"):
                self.navigate(1)
            elif low == ord("p"):
                self.navigate(-1)
            elif low == ord("j"):
                self.next_incomplete()
            elif low == ord("u"):
                self.undo()
            elif low == ord("f"):
                self.fit_view()
            elif low == ord("v"):
                self.show_overlay = not self.show_overlay
            elif low == ord("h"):
                self.show_help = not self.show_help
            elif low == ord("w"):
                self.offset_y += 60
                self.clamp_view()
            elif low == ord("a"):
                self.offset_x += 60
                self.clamp_view()
            elif low == ord("d"):
                self.offset_x -= 60
                self.clamp_view()
            elif low == ord("x"):
                self.offset_y -= 60
                self.clamp_view()

        cv2.destroyAllWindows()
        return 0


def main() -> int:
    args = parse_args()
    settings = load_settings(args)
    if args.check:
        return check_dataset(settings)
    try:
        return Annotator(settings, args.start).run()
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
