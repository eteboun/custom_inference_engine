import random

from transformers import GenerationConfig

from common import MODEL_ID


QUESTIONS = [
    "What is a GPU warp?",
    "Hi",
    "Write a haiku about the sea.",
    "Explain the difference between a process and a thread in detail, with examples.",
    "What is 2+2?",
    "List three prime numbers.",
    "Translate 'good morning' to French.",
    "Why is the sky blue? Answer in one sentence.",
    "Name the planets of the solar system.",
    "What does CUDA stand for?",
    "Summarize the plot of Romeo and Juliet.",
    "Write a Python function that reverses a linked list.",
    "What are the main causes of inflation?",
    "Give me five tips for better sleep.",
    "Explain how a transformer neural network works.",
    "What is the capital of Australia?",
    "Describe the water cycle step by step.",
    "Write a short story about a robot learning to paint.",
    "Compare TCP and UDP.",
    "What is the Pythagorean theorem? Give an example.",
    "How do vaccines work?",
    "Write a SQL query that returns the top 5 customers by revenue.",
    "What is the difference between weather and climate?",
    "Explain recursion to a ten-year-old.",
]


def load_eos_token_ids(tokenizer):
    eos = GenerationConfig.from_pretrained(MODEL_ID).eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        eos = [eos]
    return set(eos)


def build_prompts(tokenizer, n_requests, seed):
    rng = random.Random(seed)
    prompts = []
    for _ in range(n_requests):
        question = rng.choice(QUESTIONS)
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        prompts.append(list(encoded["input_ids"]))
    return prompts


def poisson_arrivals(n_requests, rate, seed):
    rng = random.Random(seed)
    t = 0.0
    arrivals = []
    for _ in range(n_requests):
        t += rng.expovariate(rate)
        arrivals.append(t)
    return arrivals
