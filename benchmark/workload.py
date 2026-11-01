import dataclasses

import numpy as np

SHORT_TOKENS = (30, 50)
LONG_TOKENS = (500, 1000)

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


CONTEXT_PARAGRAPHS = [
    "The history of computing is often told as a sequence of machines, but it is just as much a history of ideas. "
    "Early mechanical calculators could add and subtract, yet the notion of a stored program, where instructions "
    "live in the same memory as data, changed what a machine could be. Once programs could be loaded and modified "
    "like any other data, a single machine could become a word processor, a simulator or a game in an instant.",
    "Modern processors hide enormous complexity behind a simple model of sequential execution. Pipelines overlap "
    "the stages of many instructions, branch predictors guess the direction of conditional jumps, and out-of-order "
    "engines reorder work while preserving the illusion that each instruction finishes before the next begins. "
    "Caches sit between the core and main memory, because fetching a value from DRAM can take hundreds of cycles.",
    "Graphics processors took a different path. Instead of making one thread fast, they run thousands of threads "
    "at once and switch between groups of them whenever one group stalls on memory. This makes them well suited to "
    "workloads with abundant parallelism, such as shading pixels, multiplying large matrices or training neural "
    "networks, where the same operation is applied independently to many pieces of data.",
    "A river basin collects rain and snowmelt from a wide area and channels it toward the sea. Along the way the "
    "water carves valleys, deposits sediment on floodplains and supports forests, wetlands and farms. Engineers "
    "build dams to store water for dry seasons and to generate electricity, but each dam also changes the flow of "
    "sediment and the movement of fish, with consequences that may only appear decades later.",
    "Markets coordinate the decisions of millions of people who never meet. Prices carry information about scarcity: "
    "when a crop fails, its price rises, encouraging buyers to economize and sellers to bring supplies from farther "
    "away. Economists debate how well this works in practice, because information is uneven, some costs are not "
    "reflected in prices, and expectations about the future can drive prices far from underlying value.",
    "The immune system distinguishes the body's own cells from invaders using a layered defense. Physical barriers "
    "such as skin keep most pathogens out, innate cells respond quickly to common warning signs, and the adaptive "
    "system slowly learns to recognize specific molecules. Memory cells formed during an infection allow a much "
    "faster response the next time, which is the principle that vaccination relies on.",
    "Writing software that many people maintain over years requires more than correct code. Names must communicate "
    "intent, modules should have clear boundaries, and tests protect behavior when the code changes. Teams adopt "
    "conventions not because any single convention is perfect, but because a shared style lets readers focus on "
    "what the code does rather than on how it happens to be formatted.",
    "Cities grew where trade routes met: at river crossings, natural harbors and the edges of fertile plains. Their "
    "density made them engines of exchange and invention, but also of disease and fire. Over centuries, sewers, "
    "building codes, public transport and zoning reshaped urban life, and each solution introduced new trade-offs "
    "between cost, comfort, freedom and safety.",
    "Language models predict the next token in a sequence. During training they see vast amounts of text and adjust "
    "billions of parameters to make the observed continuations more likely. At inference time the model processes "
    "the prompt once, storing intermediate keys and values in a cache, and then generates one token at a time, "
    "reusing that cache so that each new step only has to process the newest token.",
    "Serving such models efficiently is a scheduling problem as much as a numerical one. Requests arrive at random "
    "times with prompts of very different lengths. A good scheduler keeps the accelerator busy by batching many "
    "requests together, splits long prompts into chunks so they do not block short ones, and manages memory for "
    "the key-value cache in fixed-size pages so that fragmentation stays low.",
    "Astronomers measure the distance to nearby stars by watching them shift slightly against the background as the "
    "Earth orbits the Sun. For more distant galaxies they rely on standard candles, objects whose true brightness is "
    "known, such as certain pulsating stars and a particular type of supernova. Comparing true and apparent "
    "brightness gives the distance, and combining distance with redshift revealed that the universe is expanding.",
    "Learning a musical instrument trains both the hands and the ear. Beginners practice scales slowly until the "
    "movements become automatic, freeing attention for phrasing and dynamics. Teachers often recommend short daily "
    "sessions over long occasional ones, because skills consolidate between sessions, and recording oneself reveals "
    "habits that are hard to notice while playing.",
]


@dataclasses.dataclass
class WorkloadInfo:
    kind: str
    n_short: int
    n_long: int
    mean_tokens: float
    seed: int


def chat_template_parts(tokenizer) -> tuple[list[int], list[int]]:
    marker = "<<CONTENT>>"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
    )
    head, tail = text.split(marker)
    return (
        tokenizer.encode(head, add_special_tokens=False),
        tokenizer.encode(tail, add_special_tokens=False),
    )


def short_prompt(tokenizer, rng: np.random.Generator) -> list[int]:
    head, tail = chat_template_parts(tokenizer)
    while True:
        question = QUESTIONS[rng.integers(len(QUESTIONS))]
        prompt = head + tokenizer.encode(question, add_special_tokens=False) + tail
        if SHORT_TOKENS[0] <= len(prompt) <= SHORT_TOKENS[1]:
            return prompt


def long_prompt(tokenizer, rng: np.random.Generator) -> list[int]:
    head, tail = chat_template_parts(tokenizer)
    question = QUESTIONS[rng.integers(len(QUESTIONS))]

    intro = tokenizer.encode(
        "Read the following text, then answer the question.\n\n",
        add_special_tokens=False
    )
    outro = tokenizer.encode(
        f"\n\nQuestion: {question}",
        add_special_tokens=False
    )

    target = int(rng.integers(LONG_TOKENS[0], LONG_TOKENS[1] + 1))
    n_context = target - len(head) - len(intro) - len(outro) - len(tail)

    context = []
    while len(context) < n_context:
        paragraphs = rng.permutation(CONTEXT_PARAGRAPHS).tolist()
        context += tokenizer.encode(
            "\n\n".join(paragraphs) + "\n\n",
            add_special_tokens=False
        )

    return head + intro + context[:n_context] + outro + tail


def build_workload(
        tokenizer,
        kind: str,
        n: int,
        seed: int
) -> tuple[list[list[int]], WorkloadInfo]:

    n_long = {"short": 0, "long": n, "mixed": n // 2}[kind]
    n_short = n - n_long

    rng = np.random.default_rng(seed)
    prompts = [short_prompt(tokenizer, rng) for _ in range(n_short)]
    prompts += [long_prompt(tokenizer, rng) for _ in range(n_long)]
    rng.shuffle(prompts)

    info = WorkloadInfo(
        kind=kind,
        n_short=n_short,
        n_long=n_long,
        mean_tokens=float(np.mean([len(prompt) for prompt in prompts])),
        seed=seed
    )

    return prompts, info


def poisson_arrivals(
        n: int,
        seed: int,
        rate: float
) -> list[float]:

    rng = np.random.default_rng(seed)
    gaps = rng.exponential(
        1 / rate,
        size=n
    )
    arrivals = np.cumsum(gaps)

    return arrivals.tolist()
