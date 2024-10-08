import cv2
import os
import shutil

def apply_canny_filter(input_image_path, output_folder, low_threshold=100, high_threshold=200):
    # Read the input image
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Failed to load image: {input_image_path}")
        return
    
    # Get the image name and extension
    image_name = os.path.basename(input_image_path)
    image_name_no_ext, ext = os.path.splitext(image_name)
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Copy the original image to the output folder
    original_copy_path = os.path.join(output_folder, image_name)
    shutil.copy(input_image_path, original_copy_path)
    
    # Apply the Canny filter
    edges = cv2.Canny(img, low_threshold, high_threshold)
    
    # Save the Canny filter result to the output folder
    output_image_path = os.path.join(output_folder, f"{image_name_no_ext}_canny{ext}")
    cv2.imwrite(output_image_path, edges)
    
    print(f"Original image copied to: {original_copy_path}")
    print(f"Canny filter result saved to: {output_image_path}")

# Example usage
input_image_path = "/home/ychen/Documents/project/Data-Project/results/0922_Duct_fiber_suspeciouse/test_55/images/testA_newAspect/O20PR00002_P000041_S00_342_rec_B.png"  # Replace with the path to your input PNG file
output_folder = "/home/ychen/Documents/project/Data-Project/results/0922_Duct_fiber_suspeciouse/test_55/images/testA_newAspect/test_55_canny_result"  # Replace with the desired output folder
apply_canny_filter(input_image_path, output_folder)
