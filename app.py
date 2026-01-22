import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from src.models.base import BaseCNN
from src.dataset import CLASS_LABELS

model, loaded_args = BaseCNN.from_checkpoint("checkpoints/resnext.ckpt", num_classes=len(CLASS_LABELS))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    image = Image.fromarray(image)
    return transform(image).unsqueeze(0)


def predict_emotion(image):
    input_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    predicted_class = np.argmax(probabilities)
    predicted_emotion = CLASS_LABELS[predicted_class]
    confidence = probabilities[predicted_class]
    return (
        predicted_emotion,
        confidence,
        {CLASS_LABELS[i]: float(probabilities[i]) for i in range(len(CLASS_LABELS))},
    )


with gr.Blocks(title="Emotion Recognition") as demo:
    gr.Markdown("# Emotion Recognition")
    gr.Markdown("Upload an image to detect the emotion.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload an image", type="numpy")
            
            examples = gr.Examples(
                examples=["sample_happiness.png", "sample_sadness.png", "sample_surprise.png"],
                inputs=input_image,
                label="Examples"
            )
            
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear")

        with gr.Column():
            output_emotion = gr.Textbox(label="Predicted Emotion")
            output_confidence = gr.Textbox(label="Confidence")
            output_probs = gr.Label(label="Probabilities")

    submit_btn.click(
        fn=predict_emotion,
        inputs=input_image,
        outputs=[output_emotion, output_confidence, output_probs],
    )
    
    clear_btn.click(
        lambda: (None, None, None, None),
        inputs=None,
        outputs=[input_image, output_emotion, output_confidence, output_probs],
        queue=False
    )

if __name__ == "__main__":
    demo.launch()
