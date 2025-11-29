from flask import Flask, render_template, request
import tensorflow as tf

# ------------------------------
# Load model
# ------------------------------
model = tf.keras.models.load_model("toxicity.h5", compile=False)

# Recreate vectorizer
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=None,
    output_sequence_length=1800,
    output_mode="int"
)

# Load vocab
with open("vectorizer_vocab.txt", "r", encoding="utf-8") as f:
    vocab = f.read().splitlines()

seen = set()
unique_vocab = []
for word in vocab:
    if word not in seen:
        unique_vocab.append(word)
        seen.add(word)

vectorizer.set_vocabulary(unique_vocab[:20000])

# Dictionary of safe replacements
safe_replacements = {
    "fucker": "person",
    "fuck": "mess up",
    "shit": "nonsense",
    "kill": "defeat",
    "stupid": "silly",
    "idiot": "friend",
    "hate": "dislike",
    "dumb": "unkind",
    "loser": "opponent",
    "bastard": "stranger",
    "ugly": "unpleasant",
    "moron": "buddy",
    "crazy": "wild",
    "fool": "silly one",
    "trash": "bad",
    "jerk": "rude person",
    "fat": "big",
    "skinny": "slim",
    "lazy": "slow",
    "nonsense": "unhelpful",
    "hell": "trouble",
    "fucked": "messed up",
    "fucking": "messed up"
}

# ------------------------------
# Flask App
# ------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    text = ""
    highlighted_text = None
    suggestions = {}

    if request.method == "POST":
        text = request.form.get("text", "")
        if not text.strip():
            prediction = "⚠ Please enter some text."
        else:
            input_text = tf.constant([text])
            input_vector = vectorizer(input_text)
            pred = model.predict(input_vector)[0][0]
            probability = round(float(pred) * 100, 2)

            if pred > 0.5:
                prediction = f"Toxic ({probability}%)"
            else:
                prediction = f"Not Toxic ({100 - probability}%)"

            # Highlight toxic words and prepare suggestions
            words = text.split()
            highlighted = []
            for word in words:
                lower_word = word.lower().strip(".,!?")
                if lower_word in safe_replacements:
                    highlighted.append(f"<span class='toxic'>{word}</span>")
                    suggestions[word] = safe_replacements[lower_word]
                else:
                    highlighted.append(word)
            highlighted_text = " ".join(highlighted)

    return render_template(
        "index.html",
        prediction=prediction,
        text=text,
        highlighted_text=highlighted_text,
        suggestions=suggestions
    )

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
