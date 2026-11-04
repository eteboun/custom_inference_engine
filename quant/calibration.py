import torch

from benchmark.workload import build_workload
from quant.core import finalize_fp8_scales


CALIBRATION_SEED = 20261104

CALIBRATION_QUESTIONS = [
    "How do bees make honey?",
    "Why is the sky blue?",
    "Give me three tips for learning a new language.",
    "What caused the fall of the Roman Empire?",
    "Write a short poem about autumn rain.",
    "How does a bicycle stay upright while moving?",
    "Summarize how vaccines train the immune system.",
    "What makes a rainbow appear after a storm?",
    "Translate the sentence 'Good morning, my friend' into Spanish.",
    "Suggest a simple recipe for lentil soup.",
    "How do tides work?",
    "Why do cats purr?",
    "List the planets of the solar system in order.",
    "What does a central bank do?",
]

CALIBRATION_PARAGRAPHS = [
    "Coral reefs are built over thousands of years by tiny animals that deposit calcium carbonate "
    "around their soft bodies. The reef provides shelter for fish, crustaceans and molluscs, and in "
    "return many of these animals keep algae from smothering the corals. When the water warms for "
    "weeks at a time, the corals expel the microscopic algae living in their tissue, turn pale and "
    "may die if conditions do not recover.",
    "The printing press spread through Europe within a few decades of its invention in the fifteenth "
    "century. Books that once took a scribe many months could now be produced in days, and their "
    "prices fell sharply. Literacy rose in the towns, pamphlets carried new religious and political "
    "ideas across borders, and rulers soon tried to control what the presses were allowed to publish.",
    "A suspension bridge carries its deck on hangers attached to two main cables, which run over tall "
    "towers and are anchored firmly in the ground at each end. Most of the load travels through the "
    "cables as tension and down the towers as compression. Engineers must also consider wind, because "
    "a long and flexible deck can begin to twist if its shape allows air to push it rhythmically.",
    "Sourdough bread rises because of a community of wild yeast and bacteria kept alive in a mixture "
    "of flour and water. The yeast produces carbon dioxide that inflates the dough, while the bacteria "
    "produce acids that give the loaf its tangy taste and help it stay fresh. Bakers feed the starter "
    "regularly and adjust the temperature to control how quickly the dough ferments.",
    "Glaciers form where more snow falls in winter than melts in summer, so that year after year the "
    "layers compress into dense ice. Under its own weight the ice slowly flows downhill, scraping "
    "valleys into broad U shapes and carrying boulders far from where they started. Scientists drill "
    "cores from ancient ice because trapped air bubbles record the composition of past atmospheres.",
    "Double-entry bookkeeping records every transaction in two accounts, once as a debit and once as a "
    "credit, so the totals on both sides should always match. Merchants in Renaissance Italy used the "
    "method to keep track of goods, loans and partnerships across long distances. A mismatch points to "
    "an error, which made it much easier to find mistakes and to judge whether a business was healthy.",
    "Migratory birds travel thousands of kilometres between breeding and wintering grounds, often "
    "returning to the same nesting site every year. They appear to combine several senses to navigate, "
    "including the position of the sun and stars, the magnetic field of the Earth and familiar "
    "landmarks. Before departure many species eat heavily and nearly double their body weight in fat.",
    "An orchestra is organized into families of instruments: strings, woodwinds, brass and percussion. "
    "The conductor sets the tempo, shapes the dynamics and cues entrances, but much of the coordination "
    "also happens between the players, who listen closely to one another. The string section usually "
    "forms the largest group, and its principal violinist leads the tuning before a performance.",
]


def calibrate(
        model,
        tokenizer,
        n_prompts: int = 16,
        max_new_tokens: int = 32,
        margin: float = 1.1
) -> None:
    prompts, _ = build_workload(
        tokenizer,
        kind="mixed",
        n=n_prompts,
        seed=CALIBRATION_SEED,
        questions=CALIBRATION_QUESTIONS,
        paragraphs=CALIBRATION_PARAGRAPHS
    )

    with torch.inference_mode():
        for prompt in prompts:
            model.generate(
                torch.tensor([prompt], device=model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id
            )

    finalize_fp8_scales(model, margin=margin)
