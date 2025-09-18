import traceback
from typing import Dict, Any, Optional
from threading import Lock
from flask import request, jsonify
from vision_agent_tools.shared_types import BaseTool, Device

# 全局共享模型管理器实例
_shared_model_manager = None
_manager_lock = Lock()
_tool_instances = {}

def get_shared_model_manager():
    """获取全局共享模型管理器实例（单例模式）"""
    global _shared_model_manager
    if _shared_model_manager is None:
        with _manager_lock:
            if _shared_model_manager is None:
                from your_module import SharedModelManager  # 根据实际路径调整
                _shared_model_manager = SharedModelManager()
    return _shared_model_manager

@app.route('/v1/tools/<tool_name>', methods=['POST'])
def tool_endpoint(tool_name: str):
    try:
        # 处理多种内容类型
        payload = {}
        files = {}
        
        if request.is_json:
            payload = request.json or {}
            files = handle_files(request.files)
        elif request.form:
            payload = request.form.to_dict()
            files = handle_files(request.files)
        else:
            payload = {}
            files = handle_files(request.files)
        
        # 分离初始化参数和调用参数
        init_params = {}
        call_params = {}
        
        init_param_keys = ["model", "device"]  # 添加device参数
        
        for key, value in payload.items():
            if key in init_param_keys:
                init_params[key] = value
            else:
                call_params[key] = value
        
        # 获取工具实例（使用SharedModelManager管理）
        tool = get_tool_instance(tool_name, config=init_params if init_params else None)
        
        # 处理文件并合并到调用参数
        call_params.update(files)
        
        # 执行工具
        result = tool(**call_params)
        return jsonify({"data": result}), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

def get_tool_instance(tool_name: str, config: Optional[Dict[str, Any]] = None) -> BaseTool:
    """获取工具实例，使用SharedModelManager管理GPU资源"""
    lock = Lock()
    
    # 构建缓存键，包含model信息
    cache_key = tool_name
    model_name = "default"
    if config and "model" in config:
        model_name = config["model"]
        cache_key = f"{tool_name}_{model_name}"
    
    with lock:
        # 检查本地缓存
        if cache_key not in _tool_instances:
            entry = TOOL_REGISTRY.get(tool_name)
            if not entry:
                raise ValueError(f"Tool '{tool_name}' not registered")
            
            tool_class = get_tool_class(tool_name)
            
            # 创建工具实例
            if config:
                # 移除device参数，因为它将通过SharedModelManager管理
                init_config = {k: v for k, v in config.items() if k != "device"}
                instance = tool_class(**init_config) if init_config else tool_class()
            else:
                instance = tool_class()
            
            # 将实例添加到SharedModelManager
            manager = get_shared_model_manager()
            model_id = manager.add(instance)
            
            # 缓存模型ID而不是实例本身
            _tool_instances[cache_key] = {
                "model_id": model_id,
                "manager": manager
            }
        
        # 从SharedModelManager获取实例
        cached_info = _tool_instances[cache_key]
        manager = cached_info["manager"]
        model_id = cached_info["model_id"]
        
        # 通过SharedModelManager获取实例（会自动处理GPU分配）
        tool_instance = manager.fetch_model(model_id)
        
        # 如果配置中指定了设备，确保设备配置正确
        if config and "device" in config:
            requested_device = Device(config["device"]) if isinstance(config["device"], str) else config["device"]
            try:
                tool_instance.to(requested_device)
            except (NotImplementedError, AttributeError):
                # 某些工具可能不支持设备切换，忽略该错误
                pass
        
        return tool_instance

# 异步版本（如果需要支持异步操作）
import asyncio

async def get_tool_instance_async(tool_name: str, config: Optional[Dict[str, Any]] = None) -> BaseTool:
    """获取工具实例的异步版本，使用SharedModelManager管理GPU资源"""
    
    # 构建缓存键
    cache_key = tool_name
    model_name = "default"
    if config and "model" in config:
        model_name = config["model"]
        cache_key = f"{tool_name}_{model_name}"
    
    # 检查本地缓存（需要线程安全）
    if cache_key not in _tool_instances:
        with _manager_lock:
            if cache_key not in _tool_instances:
                entry = TOOL_REGISTRY.get(tool_name)
                if not entry:
                    raise ValueError(f"Tool '{tool_name}' not registered")
                
                tool_class = get_tool_class(tool_name)
                
                # 创建工具实例
                if config:
                    init_config = {k: v for k, v in config.items() if k != "device"}
                    instance = tool_class(**init_config) if init_config else tool_class()
                else:
                    instance = tool_class()
                
                # 将实例添加到SharedModelManager
                manager = get_shared_model_manager()
                model_id = manager.add(instance)
                
                _tool_instances[cache_key] = {
                    "model_id": model_id,
                    "manager": manager
                }
    
    # 从SharedModelManager获取实例
    cached_info = _tool_instances[cache_key]
    manager = cached_info["manager"]
    model_id = cached_info["model_id"]
    
    # 使用信号量确保GPU独占访问
    async with manager.gpu_semaphore:
        tool_instance = manager.fetch_model(model_id)
        
        # 处理设备配置
        if config and "device" in config:
            requested_device = Device(config["device"]) if isinstance(config["device"], str) else config["device"]
            try:
                tool_instance.to(requested_device)
            except (NotImplementedError, AttributeError):
                pass
        
        return tool_instance

# 清理函数，用于释放资源
def cleanup_tool_instances():
    """清理所有工具实例和SharedModelManager"""
    global _tool_instances, _shared_model_manager
    
    with _manager_lock:
        # 清理所有模型实例
        if _shared_model_manager:
            for model_info in _tool_instances.values():
                model_id = model_info["model_id"]
                if model_id in _shared_model_manager.models:
                    # 将模型移到CPU以释放GPU内存
                    try:
                        model = _shared_model_manager.models[model_id]
                        model.to(Device.CPU)
                    except Exception as e:
                        print(f"Error moving model {model_id} to CPU: {e}")
        
        # 清空缓存
        _tool_instances.clear()
        _shared_model_manager = None

# 上下文管理器，用于自动GPU资源管理
class ToolContextManager:
    def __init__(self, tool_name: str, config: Optional[Dict[str, Any]] = None):
        self.tool_name = tool_name
        self.config = config
        self.tool_instance = None
    
    def __enter__(self):
        self.tool_instance = get_tool_instance(self.tool_name, self.config)
        return self.tool_instance
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 可以在这里添加清理逻辑，比如将模型移回CPU
        if self.tool_instance and hasattr(self.tool_instance, 'to'):
            try:
                self.tool_instance.to(Device.CPU)
            except Exception:
                pass  # 忽略清理错误

# 使用示例
def example_usage():
    # 方式1：直接使用
    tool = get_tool_instance("florence2", {"model": "microsoft/Florence-2-large", "device": "cuda"})
    result = tool(image="path/to/image.jpg", task="caption")
    
    # 方式2：使用上下文管理器
    with ToolContextManager("florence2", {"model": "microsoft/Florence-2-large"}) as tool:
        result = tool(image="path/to/image.jpg", task="caption")
    
    return result