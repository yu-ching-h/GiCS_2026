import numpy as np
from flask import Flask, render_template, jsonify, request
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

# ==========================================
# 1. 訓練 UEBA AI 模型 (小明的行為基準 Baseline)
# ==========================================
np.random.seed(42)
# 小明常規：晚上 20~22 點登入 (平均21點)，打字速度每分鐘 50~70 字 (平均60字)
normal_login_hours = np.random.normal(21, 0.5, 200)
normal_typing_speed = np.random.normal(60, 5, 200)
X_train = np.column_stack((normal_login_hours, normal_typing_speed))

# 訓練孤立森林模型
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(X_train)

# ==========================================
# 2. 網頁路由與即時分析 API
# ==========================================

@app.route('/')
def index():
    return render_template('ueba_demo.html')

@app.route('/api/stream_analyze', methods=['POST'])
def stream_analyze():
    data = request.json
    login_time = float(data.get('login_time', 21.0))
    typing_speed = float(data.get('typing_speed', 60.0))
    
    current_behavior = np.array([[login_time, typing_speed]])
    
    # 進行單次即時判定 (1 正常, -1 異常)
    prediction = clf.predict(current_behavior)[0]
    
    # 計算異常分數轉換為機率
    raw_score = clf.score_samples(current_behavior)[0]
    if prediction == -1:
        # 駭客狀態：計算出高危險機率
        anomaly_probability = round(92.0 + (abs(raw_score) * 15), 1)
        if anomaly_probability > 99.8: anomaly_probability = 99.8
        status = "danger"
        msg = "🚨 偵測到行為特徵劇烈異變！Token 被冒用！"
    else:
        # 小明狀態：計算出極低冒用機率
        anomaly_probability = round(abs(raw_score) * 3, 1)
        status = "safe"
        msg = "🟢 行為特徵完全符合使用者基準"

    return jsonify({
        "status": status,
        "probability": anomaly_probability,
        "message": msg,
        "current_data": {"time": login_time, "speed": typing_speed}
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)