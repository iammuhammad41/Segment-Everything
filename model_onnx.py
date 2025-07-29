import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from your_custom_module import custom_model_registry, CustomPredictor
from your_custom_module.utils.onnx import CustomOnnxModel

def display_masks(mask_list, ax):
    """
    Display masks on the provided axes.

    :param mask_list: List of masks, each containing segmentation data
    :param ax: Axis object where masks will be displayed
    """
    if len(mask_list) == 0:
        return
    sorted_masks = sorted(mask_list, key=lambda x: x['area'], reverse=True)
    img = np.ones((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for mask in sorted_masks:
        m = mask['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.6]])
        img[m] = color_mask
    ax.imshow(img)

def display_points(coords, labels, ax, marker_size=300):
    """
    Display the input points on the image.

    :param coords: Coordinates of the points
    :param labels: Labels of the points
    :param ax: Axis object where points will be displayed
    :param marker_size: Size of the marker
    """
    positive_points = coords[labels == 1]
    negative_points = coords[labels == 0]
    ax.scatter(positive_points[:, 0], positive_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(negative_points[:, 0], negative_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def display_bounding_box(box, ax):
    """
    Display a bounding box on the image.

    :param box: Coordinates of the bounding box
    :param ax: Axis object where bounding box will be displayed
    """
    x0, y0 = box[0], box[1]
    width, height = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), width, height, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def load_and_export_model(model_type="custom_vit", checkpoint_path="custom_model_checkpoint.pth", device="cuda"):
    """
    Load and export the SAM model to ONNX format.

    :param model_type: Type of the model (default is "custom_vit")
    :param checkpoint_path: Path to the model checkpoint
    :param device: Device to run the model on (default is "cuda")

    :returns: The ONNX model for mask generation
    """
    model = custom_model_registry[model_type](checkpoint=checkpoint_path)
    model.to(device=device)

    onnx_model_path = "custom_onnx_model.onnx"
    custom_onnx_model = CustomOnnxModel(model, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    dummy_inputs = {
        "image_embeddings": torch.randn(1, 1024, 512, 512, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, 256, 256, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }

    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with open(onnx_model_path, "wb") as f:
        torch.onnx.export(
            custom_onnx_model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            opset_version=17,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    return onnx_model_path

def load_onnx_model(onnx_model_path):
    """
    Load the ONNX model using ONNX runtime.

    :param onnx_model_path: Path to the ONNX model

    :returns: ONNX runtime session for inference
    """
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    return ort_session

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess image for model input.

    :param image_path: Path to the input image
    :param target_size: Desired size of the image after resizing

    :returns: Preprocessed image
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32)
    return image

def run_inference(onnx_session, image, point_coords, point_labels):
    """
    Run inference using the ONNX model.

    :param onnx_session: ONNX runtime session
    :param image: Input image for inference
    :param point_coords: Coordinates of points for segmentation
    :param point_labels: Labels for the points

    :returns: Generated masks from the model
    """
    image_embedding = np.random.randn(1, 1024, 512, 512).astype(np.float32)
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.zeros(1, dtype=np.float32)
    orig_im_size = np.array(image.shape[:2], dtype=np.float32)

    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": mask_input,
        "has_mask_input": has_mask_input,
        "orig_im_size": orig_im_size
    }

    masks, _, _ = onnx_session.run(None, ort_inputs)
    masks = masks > 0.5  # Threshold mask to binary

    return masks

def main(image_path, onnx_model_path):
    """
    Main function to generate masks from an image.

    :param image_path: Path to the image for segmentation
    :param onnx_model_path: Path to the ONNX model
    """
    image = preprocess_image(image_path)
    point_coords = np.array([[500, 375]])  # Example points
    point_labels = np.array([1])  # Example labels

    # Load the ONNX model
    ort_session = load_onnx_model(onnx_model_path)

    # Run inference to get masks
    masks = run_inference(ort_session, image, point_coords, point_labels)

    # Display the results
    plt.imshow(image)
    display_masks(masks, plt.gca())
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image_path = 'your_image.jpg'  # Replace with the actual image path
    onnx_model_path = 'custom_onnx_model.onnx'  # Replace with the actual ONNX model path
    main(image_path, onnx_model_path)
