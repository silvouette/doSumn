from flask import Flask, render_template, request, url_for
import pandas as pd
import main
import os
app = Flask(__name__)

@app.route('/')
@app.route('/home')
def index():
    return render_template("home.html")

@app.route("/summary", methods=["GET", "POST"])
def summ():
    if request.method == 'POST':
        if request.files:
            input = request.files['input']
    doc = input.filename
    result, wordcount, removal, classed, rm_acc, cl_acc = main.dosumn(doc)

    return render_template("summary.html", result = result, total = wordcount, name = doc,  tables_1=[removal.to_html(classes='data table table-borderless')], tables_2=[classed.to_html(classes='data table table-borderless')], rm = rm_acc, cl = cl_acc)

if __name__ == '__main__':
    app.run()