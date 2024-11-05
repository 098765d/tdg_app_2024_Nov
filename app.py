from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_from_directory,send_file,abort
import json
import os
from modules.analysis_pipeline import run_pipeline  # Import the pipeline function

# Flask application setup
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure secret key

# Route for running the analysis pipeline
@app.route('/run_analysis', methods=['GET'])
def run_analysis():
    try:
        # Run the analysis pipeline
        run_pipeline()
        
        # Return a success message
        return jsonify({'message': 'Analysis pipeline executed successfully.'}), 200
    except Exception as e:
        # If there is an error, return a failure message with the error
        return jsonify({'error': str(e)}), 500

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def login():
    # For GET request, render the login page with the program_codes
    return render_template('index.html')


@app.route('/<program_code>/model_eval/img/<path:filename>')
def serve_program_images(program_code, filename):
    # Construct the full path to the image directory
    directory = os.path.join(app.root_path, program_code, 'model_eval', 'img')
    
    # Security check: Ensure the directory exists within the allowed path
    if not os.path.exists(directory):
        abort(404)
    
    # Serve the image file
    return send_from_directory(directory, filename)


@app.route('/<program_code>/model_pred/<path:filename>')
def serve_prediction_file(program_code, filename):
    # Define the base directory where prediction files are stored
    base_dir = app.root_path  # Adjust if necessary

    # Construct the full directory path based on the program code
    directory = os.path.join(base_dir, program_code, 'model_pred')
    directory = os.path.abspath(directory)

    # Construct the absolute path to the requested file
    requested_file = os.path.abspath(os.path.join(directory, filename))

    # Security check: Ensure the requested file is within the allowed directory
    if not requested_file.startswith(directory):
        abort(403)  # Forbidden

    # Check if the file exists
    if not os.path.exists(requested_file):
        abort(404)  # Not Found

    # Serve the Excel file
    return send_from_directory(directory, filename, as_attachment=True)


@app.route('/report/<program_code>')
def report_page(program_code):
    report_filename = f"{program_code}_report.html"
    report_path = os.path.abspath(os.path.join(app.root_path, program_code, report_filename))
    
    print(f"Report Path: {report_path}")
    
    if os.path.exists(report_path):
        return send_file(report_path)
    else:
        error = f"Report file for {program_code} not found."
        return render_template('report_page.html', error=error, program_code=program_code)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
