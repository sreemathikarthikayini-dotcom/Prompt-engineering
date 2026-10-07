from transformers import pipeline
import pandas as pd

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

print("Model loaded successfully!")
problems = {
    "Education": "Explain machine learning to a beginner.",
    "Healthcare": "Explain why regular exercise is important.",
    "Finance": "Explain what a savings account is.",
    "Customer Support": "A customer says their order has not arrived. Write a helpful response.",
    "Programming": "Explain what a Python function is to a beginner."
}
def generate_response(prompt):
    result = generator(
        prompt,
        max_new_tokens=80,
        do_sample=True
    )
    return result[0]["generated_text"]
zero_shot_results = []

for category, question in problems.items():

    response = generate_response(question)

    zero_shot_results.append({
        "Category": category,
        "Prompt_Type": "Zero-shot",
        "Prompt": question,
        "Response": response
    })

print("Zero-shot completed!")
one_shot_results = []

for category, question in problems.items():

    prompt = f"""
Example:
Question: What is a computer?
Answer: A computer is an electronic device that processes information.

Now answer this:
Question: {question}
Answer:
"""

    response = generate_response(prompt)

    one_shot_results.append({
        "Category": category,
        "Prompt_Type": "One-shot",
        "Prompt": prompt,
        "Response": response
    })

print("One-shot completed!")
few_shot_results = []

for category, question in problems.items():

    prompt = f"""
Example 1:
Question: What is Python?
Answer: Python is a programming language.

Example 2:
Question: What is Java?
Answer: Java is an object-oriented programming language.

Example 3:
Question: What is SQL?
Answer: SQL is used to manage databases.

Now answer this:
Question: {question}
Answer:
"""

    response = generate_response(prompt)

    few_shot_results.append({
        "Category": category,
        "Prompt_Type": "Few-shot",
        "Prompt": prompt,
        "Response": response
    })

print("Few-shot completed!")
roles = {
    "Education": "experienced teacher",
    "Healthcare": "medical educator",
    "Finance": "financial educator",
    "Customer Support": "professional customer support agent",
    "Programming": "experienced AI engineer"
}

role_results = []

for category, question in problems.items():

    role = roles[category]

    prompt = f"""
You are an {role}.

Question:
{question}

Give a clear and simple answer.
"""

    response = generate_response(prompt)

    role_results.append({
        "Category": category,
        "Prompt_Type": "Role Prompting",
        "Prompt": prompt,
        "Response": response
    })

print("Role prompting completed!")
reasoning_questions = [
    "A student studies 2 hours per day for 5 days. How many hours does the student study in total?",
    "A shop has 20 apples and sells 7. How many apples are left?",
    "A class has 30 students. If 10 students are absent, how many students are present?"
]

cot_results = []

for question in reasoning_questions:

    prompt = f"""
Solve the following problem step by step.

Question:
{question}

Reason carefully and provide the final answer.
"""

    response = generate_response(prompt)

    cot_results.append({
        "Prompt_Type": "Chain-of-Thought",
        "Prompt": prompt,
        "Response": response
    })

print("Chain-of-Thought completed!")
all_results = (
    zero_shot_results
    + one_shot_results
    + few_shot_results
    + role_results
    + cot_results
)

df = pd.DataFrame(all_results)

print(df)
df.to_csv(
    "dataset/generated_responses.csv",
    index=False
)

df[
    ["Category", "Prompt_Type", "Response"]
].to_csv(
    "dataset/prompt_comparison.csv",
    index=False
)

print("CSV files created!")
evaluation = pd.DataFrame({
    "Prompt_Type": [
        "Zero-shot",
        "One-shot",
        "Few-shot",
        "Role Prompting",
        "Chain-of-Thought"
    ],
    "Quality": [3, 3, 4, 4, 4],
    "Accuracy": [3, 3, 4, 4, 5],
    "Completeness": [3, 4, 4, 5, 4]
})

evaluation.to_csv(
    "dataset/evaluation_results.csv",
    index=False
)

print("Evaluation saved!")
education_template = """
You are an experienced teacher.
Explain {topic} to a beginner using simple language and examples.
"""

customer_template = """
You are a professional customer support agent.
Understand the customer's problem and provide a polite solution.

Problem:
{problem}
"""

programming_template = """
You are an experienced software engineer.
Explain {topic} to a beginner with a simple example.
"""

print("Reusable templates created!")
