import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, MaskAutoGenerator

def display_masks(mask_list):
    """
    Display all generated masks overlayed on the image.
    Masks are sorted based on their area.
    
    :param mask_list: List of masks, each containing segmentation information
    """
    if len(mask_list) == 0:
        return
    sorted_masks = sorted(mask_list, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for mask in sorted_masks:
        m = mask['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def load_model(model_type="vit_h", checkpoint_path="sam_vit_h_4b8939.pth", device="cuda"):
    """
    Load the SAM model for mask generation.
    
    :param model_type: Type of SAM model (default is "vit_h")
    :param checkpoint_path: Path to the model checkpoint
    :param device: Device to run the model on (default is "cuda")
    
    :returns: MaskAutoGenerator object for mask generation
    """
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)

    return MaskAutoGenerator(sam)

def generate_masks(mask_generator, image):
    """
    Generate masks for the given image using the SAM model.
    
    :param mask_generator: The MaskAutoGenerator object
    :param image: The input image for mask generation
    
    :returns: List of masks, each containing mask information
    """
    return mask_generator.generate(image)

def plot_image_with_masks(image, masks):
    """
    Display the image with all the generated masks overlaid.
    
    :param image: Input image
    :param masks: List of masks generated for the image
    """
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    display_masks(masks)
    plt.axis('off')
    plt.show()

def main(image_path, checkpoint_path="sam_vit_h_4b8939.pth", model_type="vit_h", device="cuda"):
    """
    Main function to generate masks for an image.
    
    :param image_path: Path to the input image
    :param checkpoint_path: Path to the SAM model checkpoint (default is "sam_vit_h_4b8939.pth")
    :param model_type: Type of SAM model (default is "vit_h")
    :param device: Device to run the model on (default is "cuda")
    """
    # Read and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load SAM model
    mask_generator = load_model(model_type=model_type, checkpoint_path=checkpoint_path, device=device)

    # Generate masks
    masks = generate_masks(mask_generator, image)

    # Plot image with masks overlayed
    plot_image_with_masks(image, masks)

if __name__ == "__main__":
    # Example usage
    image_path = 'path_to_your_image.jpg'  # Replace with the path to your image
    main(image_path)
