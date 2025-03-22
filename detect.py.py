import os
import cv2
import torch
import numpy as np
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device

# Clone YOLOv7 if not already present
if not os.path.exists("yolov7"):
    print("Cloning YOLOv7 repository...")
    os.system("git clone https://github.com/WongKinYiu/yolov7.git")
    os.system("cd yolov7 && pip install -r requirements.txt")  # Install dependencies

# Function to perform object detection on an image
def detect_image(image_path, weights_path, output_path, img_size=640, conf_thres=0.5, iou_thres=0.45):
    """
    Detect objects in an image using a trained YOLOv7 model.
    
    Parameters:
    - image_path: Path to the input image.
    - weights_path: Path to the trained YOLOv7 model weights.
    - output_path: Path to save the detected image.
    - img_size: Image size for YOLOv7 (default: 640).
    - conf_thres: Confidence threshold (default: 0.5).
    - iou_thres: IoU threshold for non-max suppression (default: 0.45).
    """
    # Select device (GPU or CPU)
    device = select_device('')
    
    # Load the trained YOLOv7 model
    model = torch.load(weights_path, map_location=device)['model'].float()
    model.eval()  # Set the model to evaluation mode
    
    # Load input image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    # Preprocess the image for YOLOv7
    img_resized = letterbox(img, img_size, stride=32)[0]  # Resize while keeping aspect ratio
    img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB
    img_resized = np.ascontiguousarray(img_resized)

    # Convert to PyTorch tensor and normalize
    img_tensor = torch.from_numpy(img_resized).to(device).float()
    img_tensor /= 255.0  # Normalize pixel values to [0, 1]
    img_tensor = img_tensor[None]  # Add batch dimension
    
    # Perform inference (object detection)
    with torch.no_grad():
        pred = model(img_tensor)[0]
    
    # Apply non-max suppression
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
    
    # Draw bounding boxes on the image
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=(255, 0, 0), line_thickness=2)
    
    # Save the detected image
    cv2.imwrite(output_path, img)
    print(f"Detection complete. Output saved to {output_path}")

# Example usage
if __name__ == "__main__":
    image_path = 'path_to_your_image.jpg'  # Change to your image file
    weights_path = 'runs/train/exp/weights/best.pt'  # Path to trained YOLOv7 weights
    output_path = 'output_image.jpg'  # Save path
    
    detect_image(image_path, weights_path, output_path)
