# Stub for mediapipe.solutions.pose (legacy API).
# Enables IntelliSense for Pose and process() results.

from typing import Any, List, Optional

class PoseLandmark:
    x: float
    y: float
    z: float
    visibility: float

class NormalizedLandmarkList:
    landmark: List[PoseLandmark]

class PoseOutput:
    pose_landmarks: Optional[NormalizedLandmarkList]

class Pose:
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        **kwargs: Any,
    ) -> None: ...
    def process(self, image: Any) -> PoseOutput: ...
    def close(self) -> None: ...
