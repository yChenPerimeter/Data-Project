import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


def load_model(model_path):

    model = torch.jit.load(model_path)
    model.eval()
    return model


def preprocess_image(image_path, size=(224, 224)):  # Change size to the one your model expects
    image = Image.open(image_path).convert('RGB')  # Change 'RGB' to 'L' if grayscale
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0)


def print_model_summary(model, input_image):
    device = next(model.parameters()).device
    input_image = input_image.to(device)
    x = model.features(input_image)
    print(f"Shape after feature extraction: {x.shape}")
    x = x.view(-1, model.num_flat_features(x))
    print(f"Shape after flattening: {x.shape}")


def classify_image(model, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    input_image = preprocess_image(image_path).to(device)

    print(f"Model device: {next(model.parameters()).device}")
    print(f"Input image device: {input_image.device}")

    with torch.no_grad():  # Disable gradient calculations for inference
        output = model(input_image)
    prediction = torch.argmax(F.softmax(output, dim=1), dim=1).item()
    return prediction


def process_dataset(model_path, csv_path, input_dir, output_dir):
    model = load_model(model_path)

    # Load the CSV file
    df = pd.read_csv(csv_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, row in df.iterrows():
        unique_id = row['Unique ID']
        scan_id = row['Scan ID']
        ground_truth = row['Ground Truth']
        image_files = [f for f in os.listdir(input_dir) if unique_id in f and scan_id in f]

        for image_file in image_files:
            image_path = os.path.join(input_dir, image_file)
            prediction = classify_image(model, image_path)

            result = {
                'Unique ID': unique_id,
                'Scan ID': scan_id,
                'Ground Truth': ground_truth,
                'Predicted Label': prediction,
                'Image File': image_file
            }


            result_df = pd.DataFrame([result])
            result_df.to_csv(os.path.join(output_dir, f'result_{idx}.csv'), index=False)


            if idx == 0 and image_file == image_files[0]:
                print_model_summary(model, preprocess_image(image_path))


if __name__ == "__main__":
    model_path = "/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_19_8_04/Imgassist2_rag_1.pt"
    csv_path = "/mnt/Data4/Summer2024/RNarasimha/ALL data+csv /csv_input_files/520-00030_Test_Set.csv"
    input_dir = "/mnt/Data4/Summer2024/520-00069_DataSet/Testing/Suspicious/"
    output_dir = "/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_19_8_04/failure analysis /"
    process_dataset(model_path, csv_path, input_dir, output_dir)
