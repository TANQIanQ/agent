# 视觉工具远端部署
##  当前目录
vision_agent_tools/
├── 📁 helpers/
│   ├── 🐍 __init__.py
│   ├── 🐍 filters.py
│   └── 🐍 ixc_utils.py
│
├── 📁 models/
│   ├── 🐍 __init__.py
│   ├── 🐍 clip_media_sim.py
│   ├── 🐍 controlnet_aux.py
│   ├── 🐍 depth_anything_v2.py
│   ├── 🐍 florence2_qa.py
│   ├── 🐍 florence2_sam2.py
│   ├── 🐍 florence2.py
│   ├── 🐍 flux1.py
│   ├── 🐍 internlm_xcomposer2.py
│   ├── 🐍 model_registry.py
│   ├── 🐍 nsfw_classification.py
│   ├── 🐍 nshot_counting.py
│   ├── 🐍 owlv2.py
│   ├── 🐍 qr_reader.py
│   ├── 🐍 qwen2_vl.py
│   ├── 🐍 roberta_qa.py
│   ├── 🐍 sam2.py
│   ├── 🐍 siglip.py
│   └── 🐍 utils.py
│
├── 📁 tools/
│   ├── 🐍 __init__.py
│   ├── 🐍 depth_estimation.py
│   ├── 🐍 ocr.py
│   ├── 🐍 qr_reader.py
│   ├── 🐍 shared_model_manager.py
│   ├── 🐍 text_to_classification.py
│   ├── 🐍 text_to_instance_segmentation.py
│   └── 🐍 text_to_object_detection.py
│
├── 🐍 __init__.py
├── 🐍 shared_types.py


##  文件介绍

1. 目前有两种文件：模型文件、工具文件。其中工具文件通过vision_agent_tools\models\model_registry.py调用模型。
各类模型和工具继承如下类


class BaseMLModel:
    """
    Base class for all ML models.
    This class serves as a common interface for all ML models that can be used within tools.
    """

    def __init__(self, model: str, config: dict[str, Any] | None = None):
        self.model = model

    def __call__(self):
        raise NotImplementedError("Subclasses should implement '__call__' method.")

    def to(self, device: Device):
        raise NotImplementedError("Subclass must implement 'to' method")


class BaseTool:
    """
    Base class for all tools that wrap ML models to accomplish tool tasks.
    Tools are responsible for interfacing with one or more ML models to perform specific tasks.
    """

    def __init__(
        self,
        model: str | BaseMLModel,
    ):
        self.model = model

    def __call__(self):
        raise NotImplementedError("Subclasses should implement '__call__' method.")

    def to(self, device: Device):
        raise NotImplementedError("Subclass must implement 'to' method")


### 模型文件实例

import os

#Run this line before loading torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import os.path as osp
from typing import Any, Union

import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingV2Model
from PIL import Image
from pydantic import BaseModel

from vision_agent_tools.shared_types import BaseMLModel, Device

from .utils import CHECKPOINT_DIR, download


class DepthMap(BaseModel):
    """Represents the depth map of an image.

    Attributes:
        map (Any): HxW raw depth map of the image.
    """

    map: Any

#vision_agent_tools\models\depth_anything_v2.py
class DepthAnythingV2(BaseMLModel):
    """
    Model for depth estimation using the Depth-Anything-V2 model from the paper
    [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).

    """

    _CHECKPOINT_DIR = CHECKPOINT_DIR

    def __init__(self) -> None:
        """
        Initializes the Depth-Anything-V2 model.
        """
        if not osp.exists(self._CHECKPOINT_DIR):
            os.makedirs(self._CHECKPOINT_DIR)

        DEPTH_ANYTHING_V2_CHECKPOINT = (
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
            "depth_anything_v2_vits.pth",
        )
        # init model
        self._model = DepthAnythingV2Model(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384]
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model_checkpoint_path = download(
            url=DEPTH_ANYTHING_V2_CHECKPOINT[0],
            path=os.path.join(self._CHECKPOINT_DIR, DEPTH_ANYTHING_V2_CHECKPOINT[1]),
        )

        state_dict = torch.load(self.model_checkpoint_path, map_location="cpu")
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()

    @torch.inference_mode()
    def __call__(
        self, image: Union[str, Image.Image], grayscale: bool | None = False
    ) -> DepthMap:
        """Depth-Anything-V2 is a highly practical solution for robust monocular depth estimation.

        Args:
            image (Union[str, Image.Image, np.ndarray]): The input image for depth estimation.
                Can be a file path, a PIL Image, or a NumPy array.
            grayscale (bool, optional): Whether to return the depth map as a grayscale image.
                If True, the depth map will be normalized to the range [0, 255] and converted
                to uint8. Defaults to False.

        Returns:
            DepthMap: An object type containing a numpy array with the HxW depth map of the image.
        """
        if isinstance(image, str):
            image = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        depth_map = self._model.infer_image(image)  # HxW raw depth map

        if grayscale:
            # Normalize depth map to [0, 255] and convert to uint8
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_map_normalized = (depth_map - depth_min) / (
                depth_max - depth_min + 1e-8
            )
            depth_map = (255 * depth_map_normalized).astype(np.uint8)

        return DepthMap(map=depth_map)

    def to(self, device: Device):
        self._model.to(device=device.value)


### 工具文件实例

#vision_agent_tools\tools\depth_estimation.py
from typing import Any, Dict
from enum import Enum
from PIL import Image
from vision_agent_tools.shared_types import BaseTool
from vision_agent_tools.models.model_registry import get_model_class


class DepthEstimationModel(str, Enum):
    DEPTH_ANYTHING_V2 = "depth_anything_v2"


class DepthEstimation(BaseTool):
    def __init__(self, model: DepthEstimationModel):
        model_class = get_model_class(model_name=model)
        model_instance = model_class()
        super().__init__(model=model_instance())

    def __call__(
        self,
        image: Image.Image,
        **model_config: Dict[str, Any],
    ):
        """
        Run depth estimation detection on the image provided.

        Args:
            image (Image.Image): The input image for object detection.

        Returns:
            DepthEstimationOutput: A estimation of depth.
        """
        result = self.model(image=image, **model_config)
        return result


### 模型注册文件

import re
from typing import Dict, Type

from pydantic import BaseModel, field_validator

from vision_agent_tools.shared_types import BaseMLModel

MODELS_PATH = "vision_agent_tools.models"


class ModelRegistryEntry(BaseModel):
    model_name: str
    class_name: str

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Ensure model names are lowercase and separated by underscores."""
        if not re.match(r"^[a-z][a-z0-9_]*$", v):
            raise ValueError(
                f"Model name '{v}' must be lowercase and separated by underscores."
            )
        return v

    def model_import(self) -> Type[BaseMLModel]:
        """Lazy import for a model class."""
        module = __import__(
            f"{MODELS_PATH}.{self.model_name}", fromlist=[self.class_name]
        )
        return getattr(module, self.class_name)


MODEL_REGISTRY: Dict[str, ModelRegistryEntry] = {
    "florence2": ModelRegistryEntry(
        model_name="florence2",
        class_name="Florence2",
    ),
    "florence2sam2": ModelRegistryEntry(
        model_name="florence2_sam2",
        class_name="Florence2SAM2",
    ),
    "owlv2": ModelRegistryEntry(model_name="owlv2", class_name="Owlv2"),
    "qr_reader": ModelRegistryEntry(
        model_name="qr_reader",
        class_name="QRReader",
    ),
    "nshot_counting": ModelRegistryEntry(
        model_name="nshot_counting",
        class_name="NShotCounting",
    ),
    "nsfw_classification": ModelRegistryEntry(
        model_name="nsfw_classification",
        class_name="NSFWClassification",
    ),
    "image2pose": ModelRegistryEntry(
        model_name="image2pose",
        class_name="Image2Pose",
    ),
    "internlm_xcomposer2": ModelRegistryEntry(
        model_name="internlm_xcomposer2",
        class_name="InternLMXComposer2",
    ),
    "clip_media_sim": ModelRegistryEntry(
        model_name="clip_media_sim",
        class_name="CLIPMediaSim",
    ),
    "depth_anything_v2": ModelRegistryEntry(
        model_name="depth_anything_v2",
        class_name="DepthAnythingV2",
    ),
    "flux1": ModelRegistryEntry(model_name="flux1", class_name="Flux1"),
    "qwen2_vl": ModelRegistryEntry(
        model_name="qwen2_vl",
        class_name="Qwen2VL",
    ),
    "siglip": ModelRegistryEntry(model_name="siglip", class_name="Siglip"),
}


def get_model_class(model_name: str) -> BaseMLModel:
    """
    Retrieve a model from the registry based on the model name and task

    Args:
        model_name (str): The name of the model to retrieve

    Returns:
        BaseMLModel: An instance of the requested model

    Raises:
        ValueError: If the model is not registered.
    """

    entry = MODEL_REGISTRY.get(model_name)

    if not entry:
        raise ValueError(
            f"Model '{model_name}' is not registered in the model registry."
        )

    return entry.model_import



## 目的
本项目将部署在远端，为前端提供各类模型、工具服务，因此需要将该项目扩充为一个api服务。***kill all cver*** 前端已经完成。前端调用方式如下：

_LND_API_URL = f"{_LND_BASE_URL}/v1/agent/model"
_LND_API_URL_v2 = f"{_LND_BASE_URL}/v1/tools"

def send_task_inference_request(
    payload: Dict[str, Any],
    task_name: str,
    files: Optional[List[Tuple[Any, ...]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    is_form: bool = False,
) -> Any:
    url = f"{_LND_API_URL_v2}/{task_name}"
    vision_agent_api_key = get_vision_agent_api_key()
    headers = {
        "Authorization": f"Basic {vision_agent_api_key}",
        "X-Source": "vision_agent",
    }
    session = _create_requests_session(
        url=url,
        num_retry=3,
        headers=headers,
    )

    function_name = "unknown"
    if metadata is not None and "function_name" in metadata:
        function_name = metadata["function_name"]
    response = _call_post(url, payload, session, files, function_name, is_form)
    return response["data"] if "data" in response else response


def _create_requests_session(
    url: str, num_retry: int, headers: Dict[str, str]
) -> Session:
    """Create a requests session with retry"""
    session = Session()
    retries = Retry(
        total=num_retry,
        backoff_factor=2,
        raise_on_redirect=True,
        raise_on_status=False,
        allowed_methods=["GET", "POST", "PUT"],
        status_forcelist=[
            408,  # Request Timeout
            429,  # Too Many Requests (ie. rate limiter).
            502,  # Bad Gateway
            503,  # Service Unavailable (include cloud circuit breaker)
            504,  # Gateway Timeout
        ],
    )
    session.mount(url, HTTPAdapter(max_retries=retries if num_retry > 0 else 0))
    session.headers.update(headers)
    return session


def _call_post(
    url: str,
    payload: dict[str, Any],
    session: Session,
    files: Optional[List[Tuple[Any, ...]]] = None,
    function_name: str = "unknown",
    is_form: bool = False,
) -> Any:
    files_in_b64 = None
    if files:
        files_in_b64 = [(file[0], b64encode(file[1]).decode("utf-8")) for file in files]

    tool_call_trace = None
    try:
        if files is not None:
            response = session.post(url, data=payload, files=files)
        elif is_form:
            response = session.post(url, data=payload)
        else:
            response = session.post(url, json=payload)

        tool_call_trace_payload = (
            payload
            if "function_name" in payload
            else {**payload, **{"function_name": function_name}}
        )
        tool_call_trace = ToolCallTrace(
            endpoint_url=url,
            type="tool_call",
            request=tool_call_trace_payload,
            response={},
            error=None,
            files=files_in_b64,
        )

        if response.status_code != 200:
            tool_call_trace.error = Error(
                name="RemoteToolCallFailed",
                value=f"{response.status_code} - {response.text}",
                traceback_raw=[],
            )
            _LOGGER.error(f"Request failed: {response.status_code} {response.text}")
            raise RemoteToolCallFailed(
                function_name, response.status_code, response.text
            )

        result = response.json()
        tool_call_trace.response = result
        return result
    finally:
        if tool_call_trace is not None and should_report_tool_traces():
            trace = tool_call_trace.model_dump()
            display({MimeType.APPLICATION_JSON: trace}, raw=True)

其中send_inference_request与send_task_inference_request是为满足不同前端版本的调用。

调用案例1：
def _sam2(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    image_size: Tuple[int, ...],
    image_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    if image_bytes is None:
        image_bytes = numpy_to_bytes(image)

    files = [("images", image_bytes)]
    payload = {
        "model": "sam2",
        "bboxes": json.dumps(
            [
                {
                    "labels": [d["label"] for d in detections],
                    "bboxes": [
                        denormalize_bbox(d["bbox"], image_size) for d in detections
                    ],
                }
            ]
        ),
    }

    metadata = {"function_name": "sam2"}
    pred_detections = send_task_inference_request(
        payload, "sam2", files=files, metadata=metadata
    )
    frame = pred_detections[0]
    return_data = []
    display_data = []
    for inp_detection, detection in zip(detections, frame):
        mask = rle_decode_array(detection["mask"])
        label = detection["label"]
        bbox = normalize_bbox(detection["bounding_box"], detection["mask"]["size"])
        return_data.append(
            {
                "label": label,
                "bbox": bbox,
                "mask": mask,
                "score": inp_detection["score"],
            }
        )
        display_data.append(
            {
                "label": label,
                "bbox": detection["bounding_box"],
                "mask": detection["mask"],
                "score": inp_detection["score"],
            }
        )
    return {"files": files, "return_data": return_data, "display_data": display_data}


def sam2(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """'sam2' is a tool that can segment multiple objects given an input bounding box,
    label and score. It returns a set of masks along with the corresponding bounding
    boxes and labels.

    Parameters:
        image (np.ndarray): The image that contains multiple instances of the object.
        detections (List[Dict[str, Any]]): A list of dictionaries containing the score,
            label, and bounding box of the detected objects with normalized coordinates
            between 0 and 1 (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates
            of the top-left and xmax and ymax are the coordinates of the bottom-right of
            the bounding box.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the score, label,
            bounding box, and mask of the detected objects with normalized coordinates
            (xmin, ymin, xmax, ymax). xmin and ymin are the coordinates of the top-left
            and xmax and ymax are the coordinates of the bottom-right of the bounding box.
            The mask is binary 2D numpy array where 1 indicates the object and 0 indicates
            the background.

    Example
    -------
        >>> sam2(image, [
                {'score': 0.49, 'label': 'flower', 'bbox': [0.1, 0.11, 0.35, 0.4]},
            ])
        [
            {
                'score': 0.49,
                'label': 'flower',
                'bbox': [0.1, 0.11, 0.35, 0.4],
                'mask': array([[0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0],
                    ...,
                    [0, 0, 0, ..., 0, 0, 0],
                    [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
            },
        ]
    """
    image_size = image.shape[:2]
    ret = _sam2(image, detections, image_size)
    _display_tool_trace(
        sam2.__name__,
        {"detections": detections},
        ret["display_data"],
        ret["files"],
    )

    return ret["return_data"]  # type: ignore


案例2：
def qwen2_vl_images_vqa(prompt: str, images: List[np.ndarray]) -> str:
    """'qwen2_vl_images_vqa' is a tool that can answer any questions about arbitrary
    images including regular images or images of documents or presentations. It can be
    very useful for document QA or OCR text extraction. It returns text as an answer to
    the question.

    Parameters:
        prompt (str): The question about the document image
        images (List[np.ndarray]): The reference images used for the question

    Returns:
        str: A string which is the answer to the given prompt.

    Example
    -------
        >>> qwen2_vl_images_vqa('Give a summary of the document', images)
        'The document talks about the history of the United States of America and its...'
    """
    if isinstance(images, np.ndarray):
        images = [images]

    for image in images:
        if image.shape[0] < 1 or image.shape[1] < 1:
            raise ValueError(f"Image is empty, image shape: {image.shape}")

    files = [("images", numpy_to_bytes(image)) for image in images]
    payload = {
        "prompt": prompt,
        "model": "qwen2vl",
        "function_name": "qwen2_vl_images_vqa",
    }
    data: Dict[str, Any] = send_inference_request(
        payload, "image-to-text", files=files, v2=True
    )
    _display_tool_trace(
        qwen2_vl_images_vqa.__name__,
        payload,
        cast(str, data),
        files,
    )
    return cast(str, data)

## 要求
扩充现有的代码，完成api的实现与调用。并写一个简单的测试。