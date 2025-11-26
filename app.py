import pyodbc
import re
import unidecode
from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import pandas as pd
import webbrowser
import os
import time
from dotenv import load_dotenv
from intent_engine import predict_intent, clean_text, get_response_for_intent

load_dotenv()

app = Flask(__name__)

# ---------- Database ----------
def get_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost,1445;"
        "DATABASE=AIChatBox;"
        "UID=sa;"
        "PWD=Thanhtam123@;"
    )

def load_all_products():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, description, category, price, image FROM products")
    columns = [column[0] for column in cursor.description]
    products = [dict(zip(columns, row)) for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return products

ALL_PRODUCTS = load_all_products()

def search_products_in_memory(keyword, products=ALL_PRODUCTS):
    keyword_words = set(clean_text(keyword).split())
    results = []
    for p in products:
        fields = " ".join([
            clean_text(p.get('name','')),
            clean_text(p.get('description') or ""),
            clean_text(p.get('category') or "")
        ]).split()
        if keyword_words & set(fields):
            results.append(p)
    return results[:5]


def smart_fallback_intent(message, intent):
    """
    Nếu intent là 'unknown' thì kiểm tra lại bằng từ khóa thông minh.
    """
    if intent != "unknown":
        return intent  # đã có kết quả chắc chắn

    msg = message.lower()

    # nhóm liên quan đến sản phẩm / mua hàng
    product_keywords = [
        "mua", "bán", "sản phẩm", "giá", "bao nhiêu", 
        "shop", "quà", "lưu niệm", "ly", "tượng", "đồ trang trí", "móc khóa"
    ]

    # nhóm liên quan đến cửa hàng
    store_keywords = [
        "địa chỉ", "mở cửa", "đóng cửa", "ở đâu", 
        "liên hệ", "giờ làm việc", "cửa hàng", "map"
    ]

    if any(k in msg for k in product_keywords):
        return "find_product"
    elif any(k in msg for k in store_keywords):
        return "store_info"
    else:
        return "unknown"


# ---------- Generate reply ----------
def generate_reply(message):
    intent = predict_intent(message)
    intent = smart_fallback_intent(message, intent)

    reply = ""
    products = []

    if intent == "find_product":
        products = search_products_in_memory(message)
        if products:
            reply_lines = [get_response_for_intent("find_product") or "✨ Đây là các sản phẩm mình tìm được:"]
            for p in products:
                name = p.get('name', 'Sản phẩm')
                price = p.get('price', '?')
                desc = p.get('description', '')
                image = p.get('image', '')

                # 🖼️ Thêm ảnh nếu có (hỗ trợ URL hoặc file tĩnh)
                if image:
                    if not (image.startswith("http") or image.startswith("/")):
                        image = f"/static/{image}"  # đường dẫn nội bộ
                    img_html = f'<br><img src="{image}" alt="{name}" style="width:160px; border-radius:8px; margin:6px 0;">'
                else:
                    img_html = ""

                reply_lines.append(
                    f"<br>💎 <b>{name}</b> — {price} VND"
                    f"{img_html}"
                    f"<br>👉 {desc}"
                )

            reply = "<br>".join(reply_lines)
        else:
            reply = "Xin lỗi, mình không tìm thấy sản phẩm này trong kho hiện tại. 😊"

    elif intent == "store_info":
        reply = get_response_for_intent("store_info") or "🏪 Cửa hàng mình mở cửa từ 8h sáng đến 9h tối mỗi ngày nhé!"

    elif intent == "unknown":
        reply = "Xin lỗi, mình chỉ có thể hỗ trợ về sản phẩm và cửa hàng thôi nhé 😊"

    else:
        reply = get_response_for_intent(intent) or "Mình chưa hiểu rõ, bạn nói lại giúp nhé."

    print(f"🗣️ USER: {message}")
    print(f"🎯 INTENT: {intent}")
    print(f"💬 BOT: {reply[:100]}...\n")

    return intent, reply, products



# ---------- Routes ----------
@app.route("/chat", methods=["POST"])
def chat_post():
    data = request.get_json(force=True)
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "message is required"}), 400
    intent, reply, products = generate_reply(message)
    return jsonify({"message": message, "intent": intent, "reply": reply, "products": products})


@app.route("/chat", methods=["GET"])
def chat_get():
    message = request.args.get("message", "")
    if not message:
        return jsonify({"error": "message is required"}), 400
    intent, reply, products = generate_reply(message)
    return jsonify({"message": message, "intent": intent, "reply": reply, "products": products})


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.jpg', mimetype='image/jpeg')


@app.route("/", methods=["GET"])
def index():
    return "✅ Chatbot AI đang chạy — Gửi POST JSON {'message':'...'} tới /chat hoặc test GET /chat?message=xin%20chao"


@app.route("/ui")
def ui_chat():
    return render_template("chat.html")


# ---------- Chạy app ----------
if __name__ == "__main__":
    import threading

    def open_browser():
        time.sleep(1)
        webbrowser.open("http://127.0.0.1:5000/ui")

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(target=open_browser).start()
    app.run(host="0.0.0.0", port=5000, debug=True)
