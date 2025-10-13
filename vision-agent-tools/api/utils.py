import numpy as np
from PIL import Image
import io
import base64

def image_to_np(image_data: bytes) -> np.ndarray:
    """将字节流转换为numpy数组"""
    return np.array(Image.open(io.BytesIO(image_data)))

def np_to_image(np_array: np.ndarray) -> bytes:
    """将numpy数组转换为字节流"""
    img = Image.fromarray(np_array.astype('uint8'))
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    return byte_arr.getvalue()

def handle_files(request_files) -> Dict[str, Any]:
    """处理上传的文件"""
    file_data = {}
    for file in request_files.getlist('images'):
        file_data['image'] = image_to_np(file.read())
    
    # 处理额外参数
    for key, value in request_files.items():
        if key != 'images':
            file_data[key] = base64.b64decode(value.encode('utf-8'))
    
    return file_data