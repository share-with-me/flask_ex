import os
from flask import Flask, render_template, request, redirect, url_for , send_from_directory
from werkzeug import secure_filename

app = Flask(__name__) #Initialising the flask application

app.config['UPLOAD_FOLDER'] = 'uploads/' #Setting an upload directory

app.config['ALLOWED_EXTENSIONS'] = set(['txt','pdf','png','jpg', 'gif']) #Setting allowed extensions

def allowed_file(filename):  #Checking if the filename is of valid type depending upon the allowed extensions
	return '.' in filename and \
		filename.rsplit('.',1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
 
def index():  #Render the page which facilitates the upload of a file
	return render_template('file.html')

@app.route('/upload', methods = ['POST'])  #Route which handles the uploading of a file
def upload():
	file = request.files['file'] #GEtting name of uploaded file
	if file and allowed_file(file.filename): #If file exists and satisfies the allowed_extensions
		filename = secure_filename(file.filename) #Check documentation for deeper understanding of secure_filename. In a gist, it makes the filename safe

		#Save the file now
		file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
		#Once uploaded, the user is redirected to the uploaded_file route
		return redirect(url_for('uploaded_file', filename = filename))

#Route for the current file
@app.route('/uploads/<filename>')

def uploaded_file(filename):
	#Render the contents on the browser under the filename path
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
	app.run(
		host = "127.0.0.1",
		port = int(80),
		debug = True
		)
