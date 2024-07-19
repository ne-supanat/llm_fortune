import time
import json
import random
import ollama
import chromadb
from string import Template

total_amount = 21 + (13 * 4)


def main():
    template = Template(
        """
    Answer the question based ONLY on the following context: $context
    Question: $question
    """
    )

    context = draw()
    question = "what do I need to know today base on meaning of " + context[0]

    prompt = template.safe_substitute(context=context[1], question=question)
    output = ollama.generate(model="phi3:mini", prompt=prompt)

    print(prompt)
    print(output["response"])


def draw():
    f = open("data.json")
    data = json.load(f)

    result = random.choice(list(data.keys()))
    side = random.randint(0, 1)

    return (
        data[result]["name"] + " " + ("upright" if side else "reversed"),
        data[result]["upright" if side else "reversed"],
    )


start_time = time.time()
main()
print("\n--- %s seconds ---\n" % (time.time() - start_time))
