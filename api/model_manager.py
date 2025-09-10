from vision_agent_tools.models.model_registry import get_model_class, ModelRegistryEntry, MODEL_REGISTRY
from vision_agent_tools.shared_types import BaseMLModel
from typing import Dict, Any, Optional

_model_instances: Dict[str, BaseMLModel] = {}

def get_model_instance(model_name: str, config: Optional[Dict[str, Any]] = None) -> BaseMLModel:
    """获取单例模型实例，带线程安全锁"""
    from threading import Lock
    lock = Lock()
    
    with lock:
        if model_name not in _model_instances:
            entry = MODEL_REGISTRY.get(model_name)
            if not entry:
                raise ValueError(f"Model '{model_name}' not registered")
            
            model_class = get_model_class(model_name)
            instance = model_class(config=config)
            _model_instances[model_name] = instance
        
        # 检查设备配置
        if config and "device" in config:
            _model_instances[model_name].to(config["device"])
        
        return _model_instances[model_name]