from flask import Flask, render_template, request
# import main
import show
app = Flask(__name__)

@app.route('/')
@app.route('/home') #home page to display file selector
def index():
    return render_template("home.html")

@app.route("/summary", methods=["GET", "POST"]) #summary page to display 1st classification result, 2nd classification result, and final result.
def summ():
    if request.method == 'POST':
        if request.files:
            input = request.files['input'] #get file address
    doc = input.filename
    # result, wordcount, removal, classed, rm_acc, cl_acc = main.dosumn(doc)
    result, wordcount, removal, classed, rm_acc, cl_acc = show.dosumn(doc)

    #removal is the dataframe of 1st classification test set. classed is the dataframe of 2nd classification test set. rm_acc and cl_acc are accuracy result of both classification respectively.
    return render_template("summary.html", result = result, total = wordcount, name = doc,  tables_1=[removal.to_html(classes='data table table-borderless res-table')], tables_2=[classed.to_html(classes='data table table-borderless res-table')], rm = rm_acc, cl = cl_acc)

if __name__ == '__main__':
    app.run()