from flask import Flask, render_template, request, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and encoder (if you used one during training)

model = pickle.load(open("mushroom_classifier.pkl", "rb"))
#if you used a label encoder or one hot encoder, load it here as well.
#encoder = pickle.load(open('encoder.pkl', 'rb')) #example encoder load.

@app.route("/")
def home_page():
    return render_template("home.html")

@app.route("/predict-form")
def form_page():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict(): 
    try:
        # Extract form data (Corrected: Added missing features)
        features = [
            request.form["cap-shape"],
            request.form["cap-surface"],
            request.form["cap-color"],
            request.form["bruises"],
            request.form["odor"],
            request.form["gill-attachment"], #added
            request.form["gill-spacing"], #added
            request.form["gill-size"],
            request.form["gill-color"], #added
            request.form["stalk-shape"],
            request.form["stalk-root"],
            request.form["stalk-surface-above-ring"],
            request.form["stalk-surface-below-ring"],
            request.form["stalk-color-above-ring"],
            request.form["stalk-color-below-ring"],
            request.form["veil-type"],
            request.form["veil-color"],
            request.form["ring-number"],
            request.form["ring-type"],
            request.form["spore-print-color"],
            request.form["population"],
            request.form["habitat"],
        ]


        # assuming you used a mapping dictionary, here is an example.
        mapping = {
            'b': 0, 'c': 1, 'x': 2, 'f': 3, 'k': 4, 's': 5,
            'g': 0, 'y': 1, 's': 2,
            'n': 0, 'b': 1, 'c': 2, 'g': 3, 'r': 4, 'p': 5, 'u': 6, 'e': 7, 'w': 8, 'y': 9,
            't': 0, 'f': 1,
            'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8,
            'a': 0, 'd': 1, 'f': 2, 'n': 3,
            'c': 0, 'w': 1, 'd': 2,
            'b': 0, 'n': 1,
            'k': 0, 'n': 1, 'b': 2, 'h': 3, 'g': 4, 'r': 5, 'o': 6, 'p': 7, 'u': 8, 'e': 9, 'w': 10, 'y': 11,
            'e': 0, 't': 1,
            'b': 0, 'c': 1, 'u': 2, 'e': 3, 'z': 4, 'r': 5,
            'f': 0, 'y': 1, 'k': 2, 's': 3,
            'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8,
            'p': 0, 'u': 1,
            'n': 0, 'o': 1, 'w': 2, 'y': 3,
            'n': 0, 'o': 1, 't': 2,
            'c': 0, 'e': 1, 'f': 2, 'l': 3, 'n': 4, 'p': 5, 's': 6, 'z': 7,
            'k': 0, 'n': 1, 'b': 2, 'h': 3, 'r': 4, 'o': 5, 'u': 6, 'w': 7, 'y': 8,
            'a': 0, 'c': 1, 'n': 2, 's': 3, 'v': 4, 'y': 5,
            'g': 0, 'l': 1, 'm': 2, 'p': 3, 'u': 4, 'w': 5, 'd': 6
        }
        input_data = np.array([mapping[feature] for feature in features]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)
        result = "Poisonous 🍄" if prediction[0] == 1 else "Edible 🍽️"

        return render_template("index.html", prediction_text=f"The Mushroom is: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)