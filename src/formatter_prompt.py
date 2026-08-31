# FORMATTER_PROMPT = """For the above Original Question and the Final Answer written in full sentence.
# Extract ONLY the core answer —  NO details, NO explanation, NO restating the question.

# Output ONLY a JSON list of strings.
# - Single answer: ["45"]
# - Multiple items: ["cornstarch", "sugar", "lemon juice"]
# - STRICTLY follow the ordering sequence, if/any specified in the Original Question.
# Nothing else in your output — ONLY the JSON list.
# """
FORMATTER_PROMPT = """You will be given the query and final answer written in a full sentence.

If the answer sentence indicates the information could not be found or determined,
output exactly: ["unknown"]

Otherwise, extract ONLY the core answer terms as a JSON list of strings.
- Single answer: ["45"]
- Multiple items: ["cornstarch", "sugar", "lemon juice"]
- STRICTLY follow the Formatting Instructions in the Query if Any.
Output ONLY the JSON list, nothing else.
"""