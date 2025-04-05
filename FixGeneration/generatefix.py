from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_bug_fix(code_snippet):
    model_name = "unsloth/codellama-7b-bnb-4bit"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    prompt = (
        "### Input Code:\n" + code_snippet + "\n\n"
        "### Task:\nIdentify the bug type, severity, and error line number. Then provide a fixed version of the code.\n\n"
        "### Bug Report:\nBug Type:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=512)
    
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result


buggy_code = """
def divide(a, b):
    return a / b  # Potential division by zero error
"""

bug_report_and_fix = generate_bug_fix(buggy_code)
print(bug_report_and_fix)