import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from custom_model import model_registry, CustomPredictor
from custom_model.utils.onnx import CustomOnnxModel

def render_mask(mask, ax, random_coloring=False):
    """
    Render the given mask on the provided axes.

    :param mask: The mask to be displayed.
    :param ax: The axis object to render the mask on.
    :param random_coloring: Whether to use a random color for the mask (default is False).
    """
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_coloring else np.array([0.1, 0.6, 1.0, 0.6])
    height, width = mask.shape[-2:]
    mask_image = mask.reshape(height, width, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def plot_points(coords, labels, ax, size=375):
    """
    Plot the points on the given axes with specified labels.

    :param coords: Coordinates of the points.
    :param labels: Labels for the points (1 for foreground, 0 for background).
    :param ax: The axis object to plot on.
    :param size: The size of the plotted points (default is 375).
    """
    foreground_points = coords[labels == 1]
    background_points = coords[labels == 0]
    ax.scatter(foreground_points[:, 0], foreground_points[:, 1], color='green', marker='*', s=size, edgecolor='white', linewidth=1.25)
    ax.scatter(background_points[:, 0], background_points[:, 1], color='red', marker='*', s=size, edgecolor='white', linewidth=1.25)

def draw_box(box, ax):
    """
    Draw a bounding box on the provided axes.

    :param box: The coordinates of the bounding box in [x0, y0, x1, y1] format.
    :param ax: The axis object to draw the box on.
    """
    x0, y0 = box[0], box[1]
    width, height = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), width, height, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def initialize_model(checkpoint_path="model_checkpoint.pth", model_type="vit_h", device="cuda"):
    """
    Initialize the custom model and prepare it for segmentation tasks.

    :param checkpoint_path: Path to the model checkpoint.
    :param model_type: Type of the model to use.
    :param device: Device to run the model on (default is "cuda").

    :returns: Initialized model.
    """
    model = model_registry[model_type](checkpoint=checkpoint_path)
    model.to(device=device)
    return model

def export_onnx_model(model, dummy_inputs, output_path="model.onnx"):
    """
    Export the model to ONNX format for further use.

    :param model: The model to be exported.
    :param dummy_inputs: A dictionary of dummy inputs for the model.
    :param output_path: Path to save the exported ONNX model.
    """
    dynamic_axes = {
        "input_points": {1: "num_points"},
        "input_labels": {1: "num_points"},
    }
    output_names = ["output_masks", "iou_predictions", "low_res_masks"]

    with open(output_path, "wb") as f:
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            opset_version=17,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

def run_inference(onnx_session, image, points, labels):
    """
    Run inference on the ONNX model to generate masks.

    :param onnx_session: The ONNX runtime session.
    :param image: The input image for inference.
    :param points: The coordinates of the input points.
    :param labels: The labels of the input points.

    :returns: The predicted masks from the model.
    """
    image_embedding = np.random.randn(1, 1024, 512, 512).astype(np.float32)
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.zeros(1, dtype=np.float32)
    image_size = np.array(image.shape[:2], dtype=np.float32)

    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": points,
        "point_labels": labels,
        "mask_input": mask_input,
        "has_mask_input": has_mask_input,
        "orig_im_size": image_size
    }

    masks, _, _ = onnx_session.run(None, ort_inputs)
    return masks > 0.5

def main(image_path, onnx_model_path):
    """
    Main function to generate object masks from an image.

    :param image_path: Path to the image file.
    :param onnx_model_path: Path to the ONNX model.
    """
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    points = np.array([[500, 375], [1125, 625]])  # Example points for segmentation
    labels = np.array([1, 1])  # Example labels for the points

    # Load the ONNX model
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    # Run inference to get the masks
    masks = run_inference(ort_session, image, points, labels)

    # Display the results
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    render_mask(masks, plt.gca())
    plot_points(points, labels, plt.gca())
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image_path = 'image.jpg'  # Path to the image you want to segment
    onnx_model_path = 'custom_onnx_model.onnx'  # Path to the ONNX model
    main(image_path, onnx_model_path)
