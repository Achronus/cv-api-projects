import argparse
import json
import os
from typing import Any, Optional, Tuple

import cv2
import numpy as np

import supervision as sv

KEY_ENTER = 13  # Enter key
KEY_NEWLINE = 10  # Left click
KEY_ESCAPE = 27  # Escape key
KEY_QUIT = ord("q")  # q key
KEY_SAVE = ord("s")  # s key

THICKNESS = 2
COLORS = sv.ColorPalette.DEFAULT
WINDOW_NAME = "Draw Zones"
POLYGONS = [[]]

current_mouse_position: Optional[Tuple[int, int]] = None


def resolve_source(source_path: str) -> Optional[np.ndarray]:
    """
    Loads image or first frame of video from given path.

    Args:
        source_path (str): Path to the image or video file.

    Returns:
        Optional[np.ndarray]: Loaded image or first video frame, None if failed.
    """
    if not os.path.exists(source_path):
        return None

    image = cv2.imread(source_path)
    if image is not None:
        return image

    frame_generator = sv.get_video_frames_generator(source_path=source_path)
    frame = next(frame_generator)
    return frame


def mouse_event(event: int, x: int, y: int, flags: int, param: Any) -> None:
    """
    Handles mouse events (movement and clicks) for drawing.

    Args:
        event (int): Type of mouse event.
        x (int): X-coordinate of the mouse position.
        y (int): Y-coordinate of the mouse position.
        flags (int): Additional flags.
        param (Any): Additional parameters.
    """
    global current_mouse_position
    if event == cv2.EVENT_MOUSEMOVE:
        current_mouse_position = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        POLYGONS[-1].append((x, y))


def redraw(image: np.ndarray, original_image: np.ndarray) -> None:
    """
    Redraws the image with all polygons and current drawing.

    Args:
        image (np.ndarray): Image to draw on.
        original_image (np.ndarray): Original unmodified image.
    """
    global POLYGONS, current_mouse_position
    image[:] = original_image.copy()
    for idx, polygon in enumerate(POLYGONS):
        color = (
            COLORS.by_idx(idx).as_bgr()
            if idx < len(POLYGONS) - 1
            else sv.Color.WHITE.as_bgr()
        )

        if len(polygon) > 1:
            for i in range(1, len(polygon)):
                cv2.line(
                    img=image,
                    pt1=polygon[i - 1],
                    pt2=polygon[i],
                    color=color,
                    thickness=THICKNESS,
                )
            if idx < len(POLYGONS) - 1:
                cv2.line(
                    img=image,
                    pt1=polygon[-1],
                    pt2=polygon[0],
                    color=color,
                    thickness=THICKNESS,
                )
        if idx == len(POLYGONS) - 1 and current_mouse_position is not None and polygon:
            cv2.line(
                img=image,
                pt1=polygon[-1],
                pt2=current_mouse_position,
                color=color,
                thickness=THICKNESS,
            )
    cv2.imshow(WINDOW_NAME, image)


def close_and_finalize_polygon(image: np.ndarray, original_image: np.ndarray) -> None:
    """
    Finalizes the current polygon and starts a new one.

    Args:
        image (np.ndarray): Image to draw on.
        original_image (np.ndarray): Original unmodified image.
    """
    if len(POLYGONS[-1]) > 2:
        cv2.line(
            img=image,
            pt1=POLYGONS[-1][-1],
            pt2=POLYGONS[-1][0],
            color=COLORS.by_idx(0).as_bgr(),
            thickness=THICKNESS,
        )
    POLYGONS.append([])
    image[:] = original_image.copy()
    redraw_polygons(image)
    cv2.imshow(WINDOW_NAME, image)


def redraw_polygons(image: np.ndarray) -> None:
    """
    Redraws all completed polygons on the image.

    Args:
        image (np.ndarray): Image to draw polygons on.
    """
    for idx, polygon in enumerate(POLYGONS[:-1]):
        if len(polygon) > 1:
            color = COLORS.by_idx(idx).as_bgr()
            for i in range(len(polygon) - 1):
                cv2.line(
                    img=image,
                    pt1=polygon[i],
                    pt2=polygon[i + 1],
                    color=color,
                    thickness=THICKNESS,
                )
            cv2.line(
                img=image,
                pt1=polygon[-1],
                pt2=polygon[0],
                color=color,
                thickness=THICKNESS,
            )


def save_polygons_to_json(polygons: list[list[int]], target_path: str) -> None:
    """
    Saves the drawn polygons to a JSON file.

    Args:
        polygons (list[list[int]]): List of polygon points to save.
        target_path (str): Path to save the JSON file.
    """
    data_to_save = polygons if polygons[-1] else polygons[:-1]
    with open(target_path, "w") as f:
        json.dump(data_to_save, f)


def main(src_path: str, dest_path: str) -> None:
    """
    Main function that sets up the drawing environment and handles user input.

    Args:
        src_path (str): Path to the source image or video file.
        dest_path (str): Path to save the polygon configurations.
    """
    global current_mouse_position
    original_image = resolve_source(source_path=src_path)
    if original_image is None:
        print("Failed to load source image.")
        return

    image = original_image.copy()
    cv2.imshow(WINDOW_NAME, image)
    cv2.setMouseCallback(WINDOW_NAME, mouse_event, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == KEY_ENTER or key == KEY_NEWLINE:
            close_and_finalize_polygon(image, original_image)
        elif key == KEY_ESCAPE:
            POLYGONS[-1] = []
            current_mouse_position = None
        elif key == KEY_SAVE:
            save_polygons_to_json(POLYGONS, dest_path)
            print(f"Polygons saved to {dest_path}")
            break
        redraw(image, original_image)
        if key == KEY_QUIT:
            break

    cv2.destroyAllWindows()


def run() -> None:
    """
    Parses command-line arguments and runs the main function.

    Arguments:
        src_path (str): Path to the source image or video file.
        dest_path (str): Path to save the polygon configurations.

    Action Keys:
        Left Click: Add a point to the polygon.
        Enter: Close and finalize the current polygon.
        Esc: Clear the current polygon.
        s: Save the current polygon and start a new one.
        q: Quit the drawing process.
    """
    parser = argparse.ArgumentParser(
        description="Interactively draw polygons on images or video frames and save "
        "the annotations."
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to the source image or video file for drawing polygons.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="Path where the polygon annotations will be saved as a JSON file.",
    )
    arguments = parser.parse_args()
    main(
        src_path=arguments.src,
        dest_path=arguments.dest,
    )


if __name__ == "__main__":
    run()
