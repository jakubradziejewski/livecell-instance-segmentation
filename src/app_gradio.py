import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import sys
import os

sys.path.append(os.getcwd())

from visualize import load_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_MODEL_PATH = "models/custom_model.pth" 


def predict_single_image(image_input, model_path, score_threshold):
    """Run inference on single image and return visualization with masks."""
    if not os.path.exists(model_path):
        return image_input, f"Error: Model not found at {model_path}"
    
    try:
        model = load_model(model_path, model_type='custom', num_classes=2, device=DEVICE)
    except Exception as e:
        return image_input, f"Error loading model: {str(e)}"

    pil_image = Image.fromarray(image_input).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(pil_image).to(DEVICE)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    boxes = prediction['boxes']
    masks = prediction['masks']
    scores = prediction['scores']
    
    keep = scores > score_threshold
    boxes = boxes[keep].cpu().numpy()
    masks = masks[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(pil_image)
    ax.axis('off')

    if len(boxes) > 0:
        img_h, img_w = pil_image.size[1], pil_image.size[0]
        full_mask = np.zeros((img_h, img_w, 4))

        for idx, (mask, score) in enumerate(zip(masks, scores)):
            mask_binary = (mask.squeeze() > 0.5)
            color = plt.cm.tab20(idx % 20)
            
            full_mask[mask_binary, :3] = color[:3]
            full_mask[mask_binary, 3] = 0.5

            y_coords, x_coords = np.where(mask_binary)
            if len(y_coords) > 0:
                y_c, x_c = np.mean(y_coords), np.mean(x_coords)
                ax.text(x_c, y_c, f"{score:.2f}", color='white', 
                        fontsize=8, fontweight='bold', 
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

        ax.imshow(full_mask)

    fig.canvas.draw()
    result_img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    
    return result_img, f"Detected {len(boxes)} cells."


with gr.Blocks(title="LiveCell Inference GUI") as demo:
    gr.Markdown("# Mask R-CNN Cell Detection")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input Image")
            model_path_input = gr.Textbox(
                value=DEFAULT_MODEL_PATH, 
                label="Path to .pth model file (inside container)"
            )
            score_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.05, 
                label="Confidence Threshold"
            )
            run_btn = gr.Button("Run Detection", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(label="Prediction Result")
            output_log = gr.Textbox(label="Status")

    run_btn.click(
        fn=predict_single_image,
        inputs=[input_img, model_path_input, score_slider],
        outputs=[output_img, output_log]
    )


if __name__ == "__main__":
    print("Starting Gradio Server on port 7860...")
    demo.launch(server_name="0.0.0.0", server_port=7860)