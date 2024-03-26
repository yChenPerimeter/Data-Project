import cv2
import numpy as np

def highlight_and_annotate_image(image_path, signal_roi, noise_or_background_roi, save_path):
    """
    Draws rectangles around the signal and noise/background regions on an image, with the signal
    region in blue and the noise/background region in purple. It also annotates the image with
    the SNR formula, coloring "Signal_ROI" and "Noise_ROI" texts according to their respective
    regions, and then saves the image.

    :param image_path: Path to the image file.
    :param signal_roi: Tuple of (x1, y1, width, height) for the signal region.
    :param noise_or_background_roi: Tuple of (x1, y1, width, height) for the noise or background region.
    :param save_path: Path where the annotated image will be saved.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return

    # Define colors
    signal_color = (255, 0, 0)  # Blue for signal region
    noise_color = (255, 0, 255)  # Purple for noise region

    # Draw rectangles around the regions
    cv2.rectangle(image, (signal_roi[0], signal_roi[1]), (signal_roi[0]+signal_roi[2], signal_roi[1]+signal_roi[3]), signal_color, 2)
    cv2.rectangle(image, (noise_or_background_roi[0], noise_or_background_roi[1]), (noise_or_background_roi[0]+noise_or_background_roi[2], noise_or_background_roi[1]+noise_or_background_roi[3]), noise_color, 2)

    # Annotate with SNR formula
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    x, y = 100, 800  # Starting position of the text

    # Annotate the formula
    cv2.putText(image, "SNR = ", (x, y), font, font_scale, (255, 255, 255), thickness)
    (text_width, _), _ = cv2.getTextSize("SNR = ", font, font_scale, thickness)
    x += text_width

    cv2.putText(image, "avg(Signal_ROI)", (x, y), font, font_scale, signal_color, thickness)
    (text_width, _), _ = cv2.getTextSize("avg(Signal_ROI)", font, font_scale, thickness)
    x += text_width

    cv2.putText(image, "/", (x, y), font, font_scale, (255, 255, 255), thickness)
    (text_width, _), _ = cv2.getTextSize("/", font, font_scale, thickness)
    x += text_width

    cv2.putText(image, "std(Noise_ROI)", (x, y), font, font_scale, noise_color, thickness)
    
    # Save the image
    cv2.imwrite(save_path, image)
    print(f"Image saved to {save_path}")

def draw_regions(image_path, signal_roi, noise_or_background_roi, save_path):
    """
    Draw rectangles around the signal and noise/background regions on an image and save it.

    :param image_path: Path to the image file.
    :param signal_roi: Tuple of (x1, y1, width, height) for the signal region.
    :param noise_or_background_roi: Tuple of (x1, y1, width, height) for the noise or background region.
    :param save_path: Path where the annotated image will be saved.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return

    # Convert to a color image if it's grayscale
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw rectangles around the regions, BGR format
    cv2.rectangle(image, (signal_roi[0], signal_roi[1]), (signal_roi[0]+signal_roi[2], signal_roi[1]+signal_roi[3]), (255, 0, 0), 2)
    cv2.rectangle(image, (noise_or_background_roi[0], noise_or_background_roi[1]), (noise_or_background_roi[0]+noise_or_background_roi[2], noise_or_background_roi[1]+noise_or_background_roi[3]), (255, 0, 255), 2)

    # Save the image
    cv2.imwrite(save_path, image)
    

    print(f"Image saved to {save_path}")

if __name__ == "__main__":
    print("This script is intended to be imported and used in other scripts.")
    image_path = "/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/analysis_results/valid_thumb/paired_thumb_nail_A_1_real_B.png"
    signal_roi = (0, 65,672, 105)  # Example coordinates for the signal region
    noise_or_background_roi = (0, 520, 672, 105)  # Example coordinates for the noise/background region
    save_path = "/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/analysis_results/valid_thumb/annotated_thumb_nail_A_1_image.png"  # Specify where to save the annotated image
    save_path_formula = "/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/analysis_results/valid_thumb/annotated_thumb_nail_A_1_image_formula.png"  # Specify where to save the annotated image with formula
    # Now, instead of displaying the image, it will be saved with the regions drawn.
    draw_regions(image_path, signal_roi, noise_or_background_roi, save_path)
    highlight_and_annotate_image(image_path, signal_roi, noise_or_background_roi, save_path_formula)
