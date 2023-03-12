from flask import Flask, render_template, request
from prediction_model import prediction_model

app = Flask(__name__)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        unseen_data_original,predict,highlighted_words = prediction_model(text)
        # highlighted_text = Markup(highlighted_text)
        return render_template('result.html', original_text=unseen_data_original,Emai_Category=predict,highlighted_text=highlighted_words)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
