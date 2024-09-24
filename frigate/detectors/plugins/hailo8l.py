import logging
import cv2
import numpy as np
from typing import Dict, Any
from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig
from pydantic import Field
from typing_extensions import Literal

try:
    from hailo_platform import (
        HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
        InputVStreamParams, OutputVStreamParams, FormatType
    )
except ImportError:
    pass

logger = logging.getLogger(__name__)

DETECTOR_KEY = "hailo8l"

class HailoDetectorConfig(BaseDetectorConfig):
    type: Literal["hailo8l"]
    device: str = Field(default="PCIe", title="Device Type")
    model: Dict[str, Any] = Field(..., title="Model Configuration")

class HailoDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: HailoDetectorConfig):
        self.config = detector_config
        self.device_type = detector_config.device
        self.model_config = detector_config.model
        
        self.model_type = self.model_config.get("type")
        self.model_url = self.model_config.get("url")
        self.width = self.model_config.get("width")
        self.height = self.model_config.get("height")
        self.score_threshold = self.model_config.get("score_threshold", 0.3)
        self.max_detections = self.model_config.get("max_detections", 100)
        
        if not self.model_type or not self.model_url or not self.width or not self.height:
            raise ValueError("Model type, URL, width, and height must be provided in the configuration")

        logger.info(f"Initializing Hailo detector with model type {self.model_type}")
        self.initialize_hailo_device()

    def initialize_hailo_device(self):
        try:
            devices = Device.scan()
            self.target = VDevice(device_ids=devices)
            
            hef = HEF(self.model_url)
            configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            self.network_group = self.target.configure(hef, configure_params)[0]

            self.input_vstream_info = hef.get_input_vstream_infos()[0]
            self.output_vstream_info = hef.get_output_vstream_infos()
            
            self.input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
            self.output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)

            logger.info("Hailo device initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Hailo device: {e}")
            raise

    def detect_raw(self, tensor_input):
        logger.debug("Entering detect_raw function")

        if tensor_input is None:
            raise ValueError("The 'tensor_input' argument must be provided")

        processed_tensor = self.preprocess(tensor_input)

        try:
            with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                input_data = {self.input_vstream_info.name: np.expand_dims(processed_tensor, axis=0)}
                with self.network_group.activate():
                    infer_results = infer_pipeline.infer(input_data)

            logger.debug(f"Raw inference output: {infer_results}")

            if self.model_type == "ssd_mobilenet_v1":
                detections = self.process_ssd_mobilenet_detections(infer_results)
            elif self.model_type in ["yolov8s", "yolov6n"]:
                detections = self.process_yolo_detections(infer_results)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            return detections

        except Exception as e:
            logger.error(f"Exception during inference: {e}")
            return np.zeros((20, 6), np.float32)

    def preprocess(self, frame):
        resized_img = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized_img.astype(np.float32) / 255.0

    def process_ssd_mobilenet_detections(self, raw_detections):
        boxes = raw_detections['detection_boxes'][0]
        scores = raw_detections['detection_scores'][0]
        classes = raw_detections['detection_classes'][0]

        valid_detections = scores > self.score_threshold
        filtered_boxes = boxes[valid_detections]
        filtered_scores = scores[valid_detections]
        filtered_classes = classes[valid_detections]

        filtered_boxes = filtered_boxes[:, [1, 0, 3, 2]]  # [y1, x1, y2, x2] to [x1, y1, x2, y2]

        combined = np.column_stack((
            filtered_classes,
            filtered_scores,
            filtered_boxes
        ))

        if combined.shape[0] < 20:
            padding = np.zeros((20 - combined.shape[0], 6), dtype=np.float32)
            combined = np.vstack((combined, padding))
        else:
            combined = combined[:20]

        return combined

    def process_yolo_detections(self, raw_detections):
        detections = next(iter(raw_detections.values()))

        scores = detections[:, 4]
        valid_detections = scores > self.score_threshold
        filtered_detections = detections[valid_detections]

        sorted_indices = np.argsort(filtered_detections[:, 4])[::-1]
        filtered_detections = filtered_detections[sorted_indices]

        filtered_detections = filtered_detections[:self.max_detections]

        combined = np.column_stack((
            filtered_detections[:, 5],  # class_id
            filtered_detections[:, 4],  # confidence
            filtered_detections[:, :4]  # bounding box
        ))

        if combined.shape[0] < 20:
            padding = np.zeros((20 - combined.shape[0], 6), dtype=np.float32)
            combined = np.vstack((combined, padding))
        else:
            combined = combined[:20]

        return combined

    def __del__(self):
        if hasattr(self, 'target'):
            self.target.release()