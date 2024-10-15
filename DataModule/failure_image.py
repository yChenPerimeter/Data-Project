'''import os
import shutil
import pandas as pd

def organize_images(csv_path, input_dir, output_dir):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Ensure the output directory and subdirectories exist
    subclasses = ['DCIS-', 'DCIS+', 'IDC']
    os.makedirs(output_dir, exist_ok=True)
    for subclass in subclasses:
        os.makedirs(os.path.join(output_dir, subclass), exist_ok=True)

    for _, row in df.iterrows():
        base_image_file = row['File Name'].strip()
        subclass = row['True Class'].strip()

        if base_image_file.endswith('_lr.png'):
            possible_files = [base_image_file]
        else:
            possible_files = [
                base_image_file,
                base_image_file.replace('.png', '_lr.png')
            ]

        print(f"Checking files for base name: {base_image_file}")
        for file_name in possible_files:
            src_path = os.path.join(input_dir, file_name).strip()
            print(f"Checking file existence for: '{src_path}'")

            # Additional checks
            if not os.path.exists(src_path):
                print(f"File does not exist (os.path.exists): '{src_path}'")
            if not os.access(src_path, os.R_OK):
                print(f"No read permission for file: '{src_path}'")

            dest_dir = os.path.join(output_dir, subclass)
            dest_path = os.path.join(dest_dir, file_name).strip()

            if os.path.isfile(src_path):
                shutil.copy(src_path, dest_path)
                print(f"Copied: {src_path} to {dest_path}")
                break
            else:
                print(f"File does not exist or no read permission: '{src_path}'")


if __name__ == "__main__":
    # Paths to the CSV file, input images directory, and output directory
    csv_path = "/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_19_8_04/failure analysis /FN_subclass.csv"
    input_dir = "/mnt/Data4/Summer2024/520-00069_DataSet/Testing/Suspicious/"
    output_dir = "/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_19_8_04/failure analysis /False_Negatives_images/"

    organize_images(csv_path, input_dir, output_dir)
'''
" This code is the second part of the afilure analysis.py. It writes the physical false negative images into a folder for analysis"
import os
import shutil
import pandas as pd

def organize_images(csv_path, input_dir, output_dir):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Filter the dataframe to exclude rows where "True Class" is "not found"
    df = df[df['True Class'].isin(['DCIS-', 'DCIS+', 'IDC'])]

    # Ensure the output directory and subdirectories exist
    subclasses = ['DCIS-', 'DCIS+', 'IDC']
    os.makedirs(output_dir, exist_ok=True)
    for subclass in subclasses:
        os.makedirs(os.path.join(output_dir, subclass), exist_ok=True)

    for _, row in df.iterrows():
        base_image_file = row['File Name'].strip()
        subclass = row['True Class'].strip()

        if base_image_file.endswith('_lr.png'):
            possible_files = [base_image_file]
        else:
            possible_files = [
                base_image_file,
                base_image_file.replace('.png', '_lr.png')
            ]

        print(f"Checking files for base name: {base_image_file}")
        for file_name in possible_files:
            src_path = os.path.join(input_dir, file_name).strip()
            print(f"Checking file existence for: '{src_path}'")

            # Additional checks
            if not os.path.exists(src_path):
                print(f"File does not exist (os.path.exists): '{src_path}'")
            if not os.access(src_path, os.R_OK):
                print(f"No read permission for file: '{src_path}'")

            dest_dir = os.path.join(output_dir, subclass)
            dest_path = os.path.join(dest_dir, file_name).strip()

            if os.path.isfile(src_path):
                shutil.copy(src_path, dest_path)
                print(f"Copied: {src_path} to {dest_path}")
                break
            else:
                print(f"File does not exist or no read permission: '{src_path}'")

if __name__ == "__main__":
    # Paths to the CSV file, input images directory, and output directory
    csv_path = "/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_original_imgassist2/failure analysis /FN_subclass.csv"
    input_dir = "/mnt/Data4/Summer2024/RNarasimha/ALL data+csv /Testing_sus_nonsus/Suspicious/"
    output_dir = "/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_original_imgassist2/failure analysis /False_Negatives_images/"

    organize_images(csv_path, input_dir, output_dir)
