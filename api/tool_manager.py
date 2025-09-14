from vision_agent_tools.tools.tool_registry import get_tool_class, ToolRegistryEntry, TOOL_REGISTRY
from vision_agent_tools.shared_types import BaseTool    
from typing import Dict, Any, Optional
from vision_agent_tools.tools.shared_model_manager import  SharedModelManager

shared_model_manager = SharedModelManager()

_tool_instances: Dict[str, BaseTool] = {}

def get_tool_instance(tool_name: str, config: Optional[Dict[str, Any]] = None) -> BaseTool:
    """获取单例模型实例，带线程安全锁"""
    from threading import Lock
    lock = Lock()
    
    # 为了支持不同model的实例，需要在key中包含model信息
    cache_key = tool_name
    if config and "model" in config:
        cache_key = f"{tool_name}_{config['model']}"
    
    with lock:
        if cache_key not in _tool_instances:
            entry = TOOL_REGISTRY.get(tool_name)                                                                                                                            
            if not entry:
                raise ValueError(f"Tool '{tool_name}' not registered")
            
            tool_class = get_tool_class(tool_name)
            
            # 通用的工具实例创建：直接将config中的参数传递给__init__
            if config:
                instance = tool_class(**config)
            else:
                instance = tool_class()
            
            _tool_instances[cache_key] = instance
        
        # 检查设备配置
        if config and "device" in config:
            try:
                _tool_instances[cache_key].to(config["device"])
            except NotImplementedError:
                # 某些工具可能不支持设备切换，忽略该错误
                pass
        
        return _tool_instances[cache_key]


if __name__ == "__main__":
    print(get_tool_class("florence2"))