import gradio as gr
from typing import Dict
from transformers import pipeline

# Load pipeline globally once on CPU to avoid ZeroGPU checks and startup hangs
food_not_food_pipeline = pipeline(
    task="text-classification", 
    model="MS260000/food-not-food-text-classifier",
    device="cpu",
    top_k=None
)

def food_not_food_classifier(text: str) -> Dict[str, float]:
    outputs = food_not_food_pipeline(text)[0]
    output_dict = {}
    for item in outputs:
        output_dict[item["label"]] = item["score"]
    return output_dict

description = """
A text classifier to determine if a sentence is about food or not food. 

Fine-tuned from [DistilBERT](/MS260000/food-not-food-text-classifier) on a [small dataset of food and not food text](https://huggingface.co/MS260000/food-not-food-text-classifier).
"""

demo = gr.Interface(
    fn=food_not_food_classifier, 
    inputs="text", 
    outputs=gr.Label(num_top_classes=2),
    title="🍗🚫🥑 Food or Not Food Text Classifier",
    description=description,
    examples=[
        ["I whipped up a fresh batch of code, but it seems to have a syntax error."],
        ["A delicious photo of a plate of scrambled eggs, bacon and toast."]
    ]
)

if __name__ == "__main__":
    demo.launch()
