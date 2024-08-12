
import os
import torch
from PIL import Image
from torchvision import transforms

def load_model(model_path):
    device = torch.device('cpu')
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

def preprocess_image(image_path, size=(256, 256)):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def postprocess_output(output_tensor):
    output_image = output_tensor.squeeze(0)  # Remove batch dimension
    output_image = (output_image * 0.5 + 0.5).clamp(0, 1)  # De-normalize
    return transforms.ToPILImage()(output_image)

def save_image(image, save_path):
    image.save(save_path)

def process_images(input_dir, output_dir, model):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(('.png')):
            image_path = os.path.join(input_dir, file_name)
            input_image = preprocess_image(image_path)

            with torch.no_grad():
                output_image = model(input_image)

            output_image = postprocess_output(output_image)
            output_path = os.path.join(output_dir, file_name)
            save_image(output_image, output_path)
            print(f"Processed and saved: {output_path}")

def main():
    model_path = '/Users/ragini/Desktop/Perimeter Medical Imaging AI /My stuff /scripted model /DCIS_20240808(have flip, ImgAssistData)/Random control level 0.4/DCIS_cycleganRand_G_A_checkpoints_scriptedlatest.pt'
    input_dir = '/Users/ragini/Desktop/Perimeter Medical Imaging AI /My stuff /scripted model /scripted model output /2 DCIS -ve A_checkpoints group / input DCIS -ve without lr  '
    output_dir = '/Users/ragini/Desktop/Perimeter Medical Imaging AI /My stuff /scripted model /scripted model output /2 DCIS -ve A_checkpoints group /output '

    model = load_model(model_path)
    process_images(input_dir, output_dir, model)
if __name__ == "__main__":
    main()
