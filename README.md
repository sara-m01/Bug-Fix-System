# 🐞 Bug Detection & Automatic Bug Fix Generation using CodeLlama  


## 🚀 **Project Overview**  

🔹 This project implements an **AI-powered bug detection and fix generation system** for programming languages **(Java, Python, C++, C)** using **CodeLlama**.  
🔹 The model is **optimized with 4-bit quantization** to improve efficiency and reduce memory usage.  
🔹 A **multi-task classification model** detects bugs and generates a structured **bug report**.  
🔹 The fix generation model **automatically corrects** the detected issues and returns a **fixed version of the code**.  
🔹 A **Gradio UI** provides an interactive interface for users to **input code, detect bugs, and receive fixes**.  



## 🏗 **Project Workflow**  

    A[User Inputs Code] -->|Preprocessing| B[Bug Detection Model (CodeLlama)];
    B -->|Classifies Bug Type & Severity| C[Bug Report Generated];
    C -->|Error Line Identified| D[Bug Fix Model (CodeLlama)];
    D -->|Generates Fix| E[Fixed Code Output];
    E -->|Displayed on UI| F[User Downloads Fixed Code];


    
🎯 Key Features
✔ Bug Detection Model (Multi-Task Classification)

     Identifies bugs in code

     Determines bug type, severity, and error line number
     
✔ Bug Fix Generation

     Calls CodeLlama again to generate fixes

     Returns the corrected version of the code
     
✔ Optimized with 4-bit Quantization

     Reduces memory usage while keeping accuracy

✔ Multi-Language Support

     Works with Java, Python, C++, and C

✔ User-Friendly Interface (Gradio UI)

     Provides an interactive web-based tool

Accepts code input and displays results dynamically


🛠️ Technologies Used:

Component	Technology Stack
Model	CodeLlama (Hugging Face)
Quantization	4-bit Quantization
Programming	Python (PyTorch)
UI	Gradio
Frameworks	Transformers, Hugging Face


🔬 Model Details

    Bug Detection Model : 
    Uses CodeLlama with multi-task classification
    Predicts bug type, severity, and error line number
    Trained and quantized to 4-bit for efficiency
    
    Bug Fix Model:
    Calls CodeLlama again to generate code fixes
    Uses contextual learning to maintain code integrity



📝 Example Usage
🔹 Input Code (Python)
def divide_numbers(a, b):
    return a / b  # Possible ZeroDivisionError
print(divide_numbers(10, 0))

Output:
🔹 Bug Report
{
  "Bug Type": "Runtime Error",
  "Severity": "High",
  "Error Line": 2
}
🔹 Fixed Code
def divide_numbers(a, b):
    if b == 0:
        return "Error: Division by zero"
    return a / b
print(divide_numbers(10, 0))



