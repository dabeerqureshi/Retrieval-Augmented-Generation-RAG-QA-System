import os
import easyocr
from fpdf import FPDF

def extract_text_from_image(image_path):
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])  # Specify the language(s) to use
    result = reader.readtext(image_path)
    
    # Extract and combine text from the result
    extracted_text = " ".join([text[1] for text in result])
    return extracted_text

def extract_text_from_images_in_folder(folder_path):
    extracted_text = ""
    
    # Loop through all files in the folder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        
        # Check if the file is an image (optional, based on your file types)
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {image_name}...")
            text = extract_text_from_image(image_path)
            extracted_text += f"Text from {image_name}:\n{text}\n\n"  # Add text to output
    
    return extracted_text

def save_text_to_pdf(text, output_pdf_path):
    # Create PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", size=12)
    
    # Add text to PDF
    pdf.multi_cell(0, 10, text)
    
    # Save PDF
    pdf.output(output_pdf_path)
    print(f"PDF saved as {output_pdf_path}")

# Example usage
folder_path = "images"  # Folder containing images
output_pdf_path = "extracted_text.pdf"  # Output PDF file path

# Extract text from all images in the folder
extracted_text = extract_text_from_images_in_folder(folder_path)

# Save the extracted text into a PDF
save_text_to_pdf(extracted_text, output_pdf_path)
