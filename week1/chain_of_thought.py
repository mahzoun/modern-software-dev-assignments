import os
import re
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

# TODO: Fill this in!
YOUR_SYSTEM_PROMPT = """
solve this problem, and give the final answer on the line as "Answer: <number>"
Print all intermediate steps of thinking.

<example>
what is 123! (mod 100)? 
123! (mod 100) = (122! (mod 100) * 123 (mod 100)) mod 100
so now we need to calculate 122! (mod 100) and 123 (mod 100)
we can recursively compute 122! (mod 100) as follows:
122! (mod 100) = (121! (mod 100) * 122 (mod 100)) mod 100
and so on, until we reach the base case of 1! (mod 100)
Answer: 0
</example>

<example>
what is 3^{123} (mod 101)?
to compute a^b (mod c), we can use the method of exponentiation by squaring.
this involves breaking down the exponentiation into smaller parts and using the properties of modular arithmetic to keep the numbers manageable.
We can rewrite a^b as a^(b/2) * a^(b/2) if b is even and a^(b/2) * a^(b/2) * a if b is odd. Forexample
2^8 = 2^4 * 2^4 and 2^9 = 2^4 * 2^4 * 2
in example of 3^{123} (mod 101), we can use the same approach.
3^{123} (mod 101) = (3^{61} (mod 101) * 3^{61} (mod 101) * 3) mod 101
so now we need to calculate 3^{61} (mod 101)
we can recursively compute 3^{61} (mod 101) as follows:
3^{61} (mod 101) = (3^{30} (mod 101) * 3^{30} (mod 101) * 3 (mod 101)) mod 101
and so on, until we reach the base case of 3^{1} (mod 101)
Answer: 46
</example>
"""


USER_PROMPT = """
Solve this problem, then give the final answer on the last line as "Answer: <number>".

what is 3^{12345} (mod 100)?
"""


# For this simple example, we expect the final numeric answer only
EXPECTED_OUTPUT = "Answer: 43"


def extract_final_answer(text: str) -> str:
    """Extract the final 'Answer: ...' line from a verbose reasoning trace.

    - Finds the LAST line that starts with 'Answer:' (case-insensitive)
    - Normalizes to 'Answer: <number>' when a number is present
    - Falls back to returning the matched content if no number is detected
    """
    matches = re.findall(r"(?mi)^\s*answer\s*:\s*(.+)\s*$", text)
    if matches:
        value = matches[-1].strip()
        # Prefer a numeric normalization when possible (supports integers/decimals)
        num_match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if num_match:
            return f"Answer: {num_match.group(0)}"
        return f"Answer: {value}"
    return text.strip()


def test_your_prompt(system_prompt: str) -> bool:
    """Run up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT.

    Prints "SUCCESS" when a match is found.
    """
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.3},
        )
        output_text = response.message.content
        final_answer = extract_final_answer(output_text)
        if final_answer.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            return True
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"Actual output: {final_answer}")
    return False


if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)


