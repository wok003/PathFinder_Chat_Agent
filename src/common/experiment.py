import logging
from typing import Dict, Any
from langsmith import evaluate, Client
from orchestrator.app import BasicAgent
from common.logger_config import setup_logging
# --------------------------------------------------------------------------- #
# 1) Correctness evaluator
# --------------------------------------------------------------------------- #
def correct_label(
    inputs: Dict[str, Any],
    reference_outputs: Dict[str, Any],
    outputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Binary correctness check.

    Returns:
        {"score": 0 | 1, "key": "correct_label"}
    """
    # The key names are flexible – adjust to your dataset schema.
    logger.info(f"Input to the Evaluator, inputs: {inputs}, reference_outputs: {reference_outputs}, outputs: {outputs}")
    score = int(outputs.get("output") == reference_outputs.get("label"))
    return {"score": score, "key": "correct_label"}

def target_function(inputs: dict):
    logger.info(f"Query: {inputs}, Location: target function")
    response = agent(inputs["question"])
    logger.info(f"Final Output: {response}, Location: target function")
    return response



# --- Add Logger
logger = logging.getLogger(__name__)

client = Client()   
agent = BasicAgent()
dataset_name="GAIA_HF_Golden_Dataset"
splits = "debug_split"               #"train_L1_reasoning"
logger.info(f"Experiment: {dataset_name}, split: {splits}")

evaluate(
    target_function,
    data=client.list_examples(dataset_name=dataset_name, splits=[splits]),  # We pass in a list of Splits
    evaluators=[correct_label],
    experiment_prefix="Reasoning Examples Training split"
)