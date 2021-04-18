from flask import Flask, render_template, request, redirect, url_for
import main
import os
app = Flask(__name__)

@app.route('/')
@app.route('/home')
def index():
    return render_template("index.html")

@app.route("/summary", methods=["GET", "POST"])
def summ():
    if request.method == 'POST':
        if request.files:
            input = request.files['input']
    result = main.dosumn(input.filename)
    return render_template("index.html", data = result)

if __name__ == '__main__':
    app.run()