from projects.time_analysis.app.conf.config import SETTINGS

import numpy as np
import cv2
from ultralytics import YOLO

import supervision as sv
from supervision.annotators.base import BaseAnnotator


def annotate_frame(
    frame: np.ndarray,
    *,
    detections: sv.Detections,
    methods: list[BaseAnnotator],
    labels: list[str] | None = None,
) -> np.ndarray:
    """
    Annotates a frame using a list of supervision annotators.

    Parameters:
        frame (np.ndarray): The input video frame to annotate
        detections (sv.Detections): A supervision detections object with the detection data
        methods (list[sv.BaseAnnotator]): A list of supervision annotator instances to apply to the frame
        labels (list[str], optional): The labels for the detections. Required if `sv.LabelAnnotator` is used
    """
    frame = frame.copy()

    la_found = any(isinstance(method, sv.LabelAnnotator) for method in methods)
    if (labels is None and la_found) or (labels and not la_found):
        raise ValueError(
            "'labels' and 'sv.LabelAnnotator' must be used together, one is missing."
        )

    for method in methods:
        if isinstance(method, sv.LabelAnnotator):
            frame = method.annotate(frame, detections=detections, labels=labels)
        else:
            frame = method.annotate(frame, detections=detections)

    return frame


def main() -> None:
    model = YOLO(SETTINGS.MODEL.PATH)

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()
    mask_annotator = sv.MaskAnnotator()

    methods = [box_annotator, label_annotator]

    frame_generator = sv.get_video_frames_generator(source_path=SETTINGS.VIDEO.SRC_FILE)

    def process_frame(
        model: YOLO, frame: np.ndarray, methods: list[BaseAnnotator]
    ) -> np.ndarray:
        results: YOLO = model(
            frame,
            conf=SETTINGS.THRESHOLD.CONFIDENCE,
            iou=SETTINGS.THRESHOLD.IOU,
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]
        detections = tracker.update_with_detections(detections)

        labels = [
            f"#{tracker_id} {results.names[class_id]}"
            for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
        ]

        return annotate_frame(
            frame,
            detections=detections,
            methods=methods,
            labels=labels,
        )

    for frame in frame_generator:
        frame = process_frame(model, frame, methods)
        frame = cv2.resize(frame, (SETTINGS.VIDEO.WIDTH, SETTINGS.VIDEO.HEIGHT))

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
