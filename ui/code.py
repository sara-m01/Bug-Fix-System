import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model_name = "unsloth/codellama-7b-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")


def generate_bug_fix(code_snippet, language):
    prompt = (
        f"### Input Code ({language}):\n{code_snippet}\n\n"
        "### Task:\nIdentify the bug type, severity, and error line number. Then provide a fixed version of the code.\n\n"
        "### Bug Report:\nBug Type:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=512)
    
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result


interface = gr.Interface(
    fn=generate_bug_fix,
    inputs=[
        gr.Textbox(label="Paste Your Code Here", lines=10),
        gr.Radio(["Python", "Java", "C++", "C"], label="Select Programming Language", value="Python")
    ],
    outputs=gr.Textbox(label="Bug Fix Report"),
    title="🚀 AI Bug Detector & Fix Generator",
    description="🔍 Paste your code and get an AI-generated bug report with fixes!"
)


interface.launch()
