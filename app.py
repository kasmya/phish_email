from flask import Flask, render_template, request

from src.inference import PhishingInferenceService
from src.preprocessing import clean_text

app = Flask(__name__)

EXAMPLES = [
    "Invoice #1234\nDear Customer,\nYour invoice is ready. Download the attached file and verify your account now.",
    "Hi team,\nPlease find the project update attached. Let me know if you need anything else.\nRegards,\nOperations Team",
    "Urgent action required: your mailbox storage has exceeded its limit. Click the link below to avoid suspension.",
]


def get_service():
    if "service" not in app.config:
        app.config["service"] = PhishingInferenceService()
    return app.config["service"]


@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    error = None
    email_text = ""

    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()
        if not email_text:
            error = "Please provide email text for analysis."
        else:
            try:
                service = get_service()
                results = service.predict(clean_text(email_text))
            except Exception as exc:
                error = str(exc)

    return render_template(
        "index.html",
        examples=EXAMPLES,
        email_text=email_text,
        results=results,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
