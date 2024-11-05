import os
import json
import base64  # Import base64 for encoding images
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

def encode_image_to_base64(image_path):
    """Encode an image to Base64."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        print(f"Image file not found: {image_path}")
        return None

def generate_report(program_code, base_folder):
    # Path to the JSON file
    prog_name_file_path = './static/prog_name.json'
    
    # Read the program name abbreviation from the JSON file
    with open(prog_name_file_path, 'r') as file:
        prog_name = json.load(file)
    
    # Lookup the program name using the program code
    if program_code in prog_name:
        program_name_abb = prog_name[program_code]
        print(f"The program abbreviation for {program_code} is: {program_name_abb}")
    else:
        print(f"Program code {program_code} not found in the list.")
        return  # Exit if the program code is not found

    # Define paths to program-specific folders and files
    program_folder = os.path.join(base_folder, program_code)
    eval_summary_path = os.path.join(program_folder, 'model_eval', 'eval_summary.json')
    corr_course_path = os.path.join(program_folder, 'model_eval', 'corr_course.json')
    pred_summary_path = os.path.join(program_folder, 'model_pred', 'pred_summary.json')
    prediction_file_path = os.path.join(program_folder, 'model_pred', f'{program_code}_predictions.xlsx')
    
    # Image paths
    confusion_matrix_img_path = os.path.join(program_folder, 'model_eval', 'img', 'confusion_matrix.png')
    feature_importance_img_path = os.path.join(program_folder, 'model_eval', 'img', 'feature_importance.png')
    scatter_plot_img_path = os.path.join(program_folder, 'model_eval', 'img', 'scatter_plot.png')
    
    # Check if the necessary files exist
    if not os.path.exists(eval_summary_path) or not os.path.exists(pred_summary_path):
        print(f"Missing evaluation or prediction summary for {program_code}. Skipping report generation.")
        return

    # Load JSON data
    with open(eval_summary_path, 'r') as f:
        evaluation_summary = json.load(f)
    
    with open(corr_course_path, 'r') as f:
            correlated_courses = json.load(f)

    with open(pred_summary_path, 'r') as f:
        prediction_summary = json.load(f)

    # Get current date as the report date
    report_date = datetime.now().strftime('%Y-%m-%d')

    # Encode images to Base64
    confusion_matrix_img_base64 = encode_image_to_base64(confusion_matrix_img_path)
    feature_importance_img_base64 = encode_image_to_base64(feature_importance_img_path)
    scatter_plot_img_base64 = encode_image_to_base64(scatter_plot_img_path)

    # Set up Jinja2 environment with the templates folder
    env = Environment(loader=FileSystemLoader('templates'))  # Reference the templates folder
    template = env.get_template('report_template.html')

    # Render the template with data
    rendered_html = template.render(
        program_code=program_code,
        program_name_abb=program_name_abb,
        evaluation_summary=evaluation_summary,
        correlated_courses=correlated_courses, 
        prediction_summary=prediction_summary,
        confusion_matrix_img_base64=confusion_matrix_img_base64,
        feature_importance_img_base64=feature_importance_img_base64,
        scatter_plot_img_base64=scatter_plot_img_base64,
        prediction_file_path=prediction_file_path,
        report_date=report_date,
    )

    # Save the rendered HTML to a file within the program folder
    report_file_path = os.path.join(program_folder, f'{program_code}_report.html')
    with open(report_file_path, 'w') as report_file:
        report_file.write(rendered_html)
    
    print(f'Report generated for {program_code}: {report_file_path}')

# Example usage:
# generate_report('A4B067-Chinese', 'TDG_PROJECT')
