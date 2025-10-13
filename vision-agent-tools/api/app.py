from flask import Flask, request, jsonify
from model_manager import get_model_instance
from tool_manager import get_tool_instance
from utils import handle_files
import traceback



app = Flask(__name__)


@app.route('/v1/tools/<tool_name>', methods=['POST'])
def tool_endpoint(tool_name: str):
    try:
        # 处理多种内容类型
        payload = {}
        files = {}
        
        if request.is_json:
            # 处理JSON数据
            payload = request.json or {}
            files = handle_files(request.files)  # 即使有JSON也可能附带文件
        elif request.form:
            # 处理表单数据
            payload = request.form.to_dict()
            files = handle_files(request.files)
        else:
            # 处理原始数据或其他类型
            payload = {}
            files = handle_files(request.files)
        
        # 分离初始化参数和调用参数
        init_params = {}
        call_params = {}
        
        # 提取用于__init__的参数
        init_param_keys = ["model"]  # 可以根据需要扩展
        
        for key, value in payload.items():
            if key in init_param_keys:
                init_params[key] = value
            else:
                call_params[key] = value
        
        # 获取工具实例
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


@app.route('/v1/agent/model/<model_name>', methods=['POST'])
def model_endpoint(model_name: str):
    try:
        # 处理输入数据
        payload = request.json if request.json else {}
        files = handle_files(request.files)
        config = payload.get("config", {})
        
        # 获取模型实例
        model = get_model_instance(model_name, config)
        
        # 执行模型
        result = model(**files, **payload)
        return jsonify({"data": result}), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)