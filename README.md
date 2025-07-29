# Segmentation Tools

A minimal collection of scripts to run and export a Segment Anything–based model and generate masks.

## Files

- **predict.py**  
  Load a custom PyTorch model, export to ONNX, run inference on an image, and display masks/points/boxes.

- **model_onnx.py**  
  Load your custom SAM variant, export it to ONNX, and provide helper functions to preprocess images and run ONNX inference.

- **mask_generator.py**  
  Use Meta’s Segment Anything `MaskAutoGenerator` to produce and overlay masks on any image.

## Requirements

```bash
pip install numpy torch matplotlib opencv-python onnxruntime segment_anything
````

(Plus your own `custom_model` or `your_custom_module` packages.)

## Usage

1. **Export and run inference with `predict.py`:**

   ```bash
   python predict.py <image.jpg> <custom_onnx_model.onnx>
   ```

2. **Export your PyTorch model to ONNX with `model_onnx.py`:**

   ```bash
   # in Python REPL or another script
   from model_onnx import load_and_export_model
   onnx_path = load_and_export_model(
     model_type="custom_vit",
     checkpoint_path="custom_model_checkpoint.pth"
   )
   ```

3. **Generate masks with the official SAM generator:**

   ```bash
   python mask_generator.py path/your_image.jpg
   ```
 4. **References*
    '''
    [https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb](https://github.com/facebookresearch/segment-anything/tree/main)
    https://github.com/facebookresearch/segment-anything/tree/main
    https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once
    
    ''' 
