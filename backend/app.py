# ==========================================
# 毕设后端：SOTA Fusion 适配版 (EfficientNet-V2-L)
# 对应训练代码版本：SOTA Fusion - 稳定极速版
# ==========================================

import os
import io
import sys
import traceback

# 尝试导入依赖
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    from PIL import Image
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError as e:
    print(f"❌ 依赖缺失: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 

# ==========================================
# 核心配置
# ==========================================
# 请确保服务器上的文件名与此一致，或者修改这里
MODEL_PATH = 'hd_effnetv2_l_best.pth' 
DEVICE = torch.device('cpu')

# 全局变量
model = None
model_name = "Unknown"

def try_load_sota_model(state_dict):
    """
    尝试构建 SOTA 训练代码定义的复杂模型架构
    架构特征: EfficientNet-V2-L + 1024层 + SiLU + Dropout
    """
    print("🔄 尝试加载架构: SOTA EfficientNet-V2-L (Custom Head)...")
    
    # 1. 初始化骨干网
    net = models.efficientnet_v2_l(weights=None)
    
    # 2. 获取原始输入维度 (通常是 1280)
    num_ftrs = net.classifier[-1].in_features
    
    # 3. [关键] 重建与训练代码完全一致的分类头
    # 训练代码原文:
    # model.classifier = nn.Sequential(
    #     nn.Dropout(p=0.4),
    #     nn.Linear(num_ftrs, 1024),
    #     nn.SiLU(),
    #     nn.Dropout(p=0.3),
    #     nn.Linear(1024, 2)
    # )
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, 1024),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(1024, 2)
    )
    
    # 4. 加载权重
    net.load_state_dict(state_dict)
    return net

def get_model():
    global model, model_name
    if model is not None:
        return model
    
    print(f"⏳ [系统] 开始加载模型文件: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")

    try:
        # 读取权重
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # 构建模型
        model = try_load_sota_model(state_dict)
        model_name = "SOTA-EffNetV2-L"
        print("✅ 成功匹配架构: SOTA EfficientNet-V2-L")

        model.eval()
        return model
        
    except Exception as e:
        print(f"❌ 致命错误: {traceback.format_exc()}")
        raise e

# ==========================================
# 路由
# ==========================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'loaded_model': model_name})

@app.route('/api/predict', methods=['POST'])
def predict():
    global model
    
    try:
        if model is None:
            model = get_model()
            
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        
        # 读取图片
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # [关键] 预处理必须与训练代码一致
        # 训练代码: val_transform = A.Compose([A.Resize(224), A.Normalize(mean=[0.485...], std=[0.229...])])
        transform = transforms.Compose([
            transforms.Resize((224, 224)), # 训练代码 INPUT_SIZE = 224
            transforms.ToTensor(),
            # 使用 ImageNet 标准均值方差 (训练代码用的是这个，不是 0.5)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        tensor = transform(image).unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # 训练代码中: Label 0=Real, 1=Fake (通过 RobustDataset 逻辑推断)
            # 假定 fake_keys 对应 label 1
            fake_prob = probs[0][1].item()
            is_fake = fake_prob > 0.5
            confidence = fake_prob if is_fake else (1 - fake_prob)
            
            return jsonify({
                'label': 'AIGC Fake' if is_fake else 'Real Photo',
                'is_fake': is_fake,
                'confidence': float(confidence),
                'score': float(fake_prob),
                'model_used': model_name
            })
            
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"请求处理失败: {error_msg}")
        return jsonify({'error': '后端报错', 'details': str(e), 'trace': error_msg}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)