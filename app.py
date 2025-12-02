from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

from chatbot_core.rag_pipeline import answer_question, log_feedback

app = Flask(__name__)
CORS(app)  

@app.route("/", methods=["GET"])
def home():
    """
    Landing page. If you do not have templates/index.html,
    you can safely redirect to /chat instead.
    """
    try:
        return render_template("index.html")
    except:
        return "<h2>TA Chatbot is running! Go to <a href='/chat'>/chat</a></h2>"


@app.route("/chat", methods=["GET", "POST"])
def chat():
    """
    GET -> returns chat.html UI
    POST -> receives question + first guess and returns clean answer JSON
    """
    if request.method == "GET":
        return render_template("chat.html")

    # POST
    user_question = request.form.get("question", "").strip()
    first_guess = request.form.get("first_guess", "").strip()
    onboarding_pref = request.form.get("onboarding_pref", "concise").strip()

    result = answer_question(
        question=user_question,
        first_guess=first_guess,
        onboarding_pref=onboarding_pref,
    )

    return jsonify(result)


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Saves student feedback as JSONL in chatbot_core/logs/.
    Body JSON example:
    {
      "rating": "helpful" | "needs_work" | "confusing",
      "question": "...",
      "answer": "...",
      "first_guess": "...",
      "comment": "optional"
    }
    """
    try:
        data = request.get_json(force=True)
        log_feedback(data)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    # Visit http://127.0.0.1:5000/chat
    app.run(debug=True, host="127.0.0.1", port=5000)
