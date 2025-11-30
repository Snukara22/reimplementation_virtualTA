from flask import Flask, render_template, request, jsonify
from chatbot_core.rag_pipeline import answer_question, log_feedback

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return render_template("chat.html")
    else:
        user_question = request.form.get("question", "")
        first_guess = request.form.get("first_guess", "")
        onboarding_pref = request.form.get("onboarding_pref", "concise")

        result = answer_question(
            question=user_question,
            first_guess=first_guess,
            onboarding_pref=onboarding_pref,
        )
        return jsonify(result)


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Receive feedback for a specific answer and log it.
    Body JSON example:
    {
      "rating": "helpful" | "needs_work" | "confusing",
      "question": "...",
      "answer": "...",
      "comment": "optional free-text"
    }
    """
    data = request.get_json(force=True) or {}
    log_feedback(data)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
