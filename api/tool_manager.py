from vision_agent_tools.tools.tool_registry import get_tool_class, ToolRegistryEntry, TOOL_REGISTRY
from vision_agent_tools.shared_types import BaseTool    
from typing import Dict, Any, Optional


_tool_instances: Dict[str, BaseTool] = {}

def get_tool_instance(tool_name: str, config: Optional[Dict[str, Any]] = None) -> BaseTool:
    """获取单例模型实例，带线程安全锁"""
    from threading import Lock
    lock = Lock()
    
    with lock:
        if tool_name not in _tool_instances:
            entry = TOOL_REGISTRY.get(tool_name)                                                                                                                            
            if not entry:
                raise ValueError(f"Tool '{tool_name}' not registered")
            
            tool_class = get_tool_class(tool_name)
            instance = tool_class(config=config)
            _tool_instances[tool_name] = instance
        
        # 检查设备配置
        if config and "device" in config:
            _tool_instances[tool_name].to(config["device"])
        
        return _tool_instances[tool_name]
