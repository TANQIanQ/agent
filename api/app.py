from flask import Flask, request, jsonify
from model_manager import get_model_instance
from tool_manager import get_tool_instance

from utils import handle_files
import traceback

app = Flask(__name__)

@app.route('/v1/tools/<tool_name>', methods=['POST'])
def tool_endpoint(tool_name: str):
    try:
        # 处理输入数据
        payload = request.json if request.json else {}
        files = handle_files(request.files)
        
        # 获取工具处理器
        tool = get_tool_instance(tool_name)
        
        # 执行工具
        result = tool(**payload, **files)
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