import os
import fitz  # PyMuPDF

# Create 'images' folder if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

def extract_images_from_pdf(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    image_count = 0
    for page_num in range(len(doc)):
        # Get the page
        page = doc.load_page(page_num)
        
        # Get images on the page
        img_list = page.get_images(full=True)
        
        for img_index, img in enumerate(img_list):
            xref = img[0]  # Image reference
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]  # Image data
            
            # Define image file path
            image_filename = f"images/{os.path.basename(pdf_path).replace('.pdf', '')}_page{page_num + 1}_img{img_index + 1}.png"
            
            # Save image
            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)
                
            image_count += 1
            print(f"Image {image_count} saved as {image_filename}")

    print(f"Total {image_count} images extracted from {pdf_path}.")

# Process all PDFs in the 'data' folder
pdf_folder = 'data'
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    extract_images_from_pdf(pdf_path)
