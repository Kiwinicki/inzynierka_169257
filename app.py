import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from src.models import PlainCNN, ResNet
from src.dataset import CLASS_LABELS

model = ResNet(base_ch=32, n_classes=len(CLASS_LABELS))
model.load_state_dict(torch.load("checkpoints/resnet-ep17-loss0.388-final.ckpt"))
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


iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(label="Upload an image"),
    outputs=[
        gr.Textbox(label="Predicted Emotion"),
        gr.Textbox(label="Confidence"),
        gr.Label(label="Probabilities"),
    ],
    title="Emotion Recognition",
    description="Upload an image to detect the emotion.",
)

if __name__ == "__main__":
    iface.launch()
