
from flask import Flask, render_template, url_for, redirect, request,send_from_directory,flash, send_file
import json
from seg import *
from main import getPrediction
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField

import urllib.request
import os
from wtforms.validators import InputRequired

#from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
# from flask_bootstrap import Bootstrap
UPLOAD_FOLDER = 'static/brainimages/'
app = Flask(__name__, template_folder='templates')
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
@app.route('/homePage')
def homePage():
    return render_template("index.html", title="homePage", custom_css="index", t="index")

@app.route('/classification', methods=['POST', 'GET'])
def classification():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            # Use this werkzeug method to secure filename.
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # getPrediction(filename)
            label = getPrediction(filename)
            print(label)
            arr_list = label.tolist()

            # convert the list to a JSON string
            json_str = json.dumps(arr_list)
            flash(json_str)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            flash(full_filename)
            return redirect('/classification')

    # This code will execute for GET requests to /classification
    return render_template("classification.html", title="classification", custom_css="classification")

@app.route('/segmentation',methods=['POST', 'GET'])
def segmentation():
    if request.method=='POST':
        n1=request.form['num1']
        n2=request.form['num2']
        n3=request.form['num3']
        case=n1+n2+n3
        print(case)
        p=showPredictsByCase(case)
        return send_file(p, mimetype='image/png')

    return render_template("segmentation.html", title="segmentation", custom_css="segmentation")


@app.route('/FAQ')
def FAQ():
    return render_template("faq.html", title="FAQ", custom_css="faq")


@app.route('/information')
def information():
    return render_template("info.html", title="information", custom_css="info")


@app.route('/Stories')
def Stories():
    return render_template("stories.html", title="Stories", custom_css="stories")


@app.route('/memory_game')
def memory_game():
    return render_template("memory_game.html", title="memory_game", custom_css="memory_game")


@app.route('/treatment')
def treatment():
    return render_template("treatment.html", title="Treatment", custom_css="treatment")


@app.route('/about_US')
def about_US():
    return render_template("about_US.html", title="AboutUS", custom_css="about_US")

if __name__ == '__main__':
    app.run(debug=True)
