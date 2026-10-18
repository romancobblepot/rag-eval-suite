from dotenv import load_dotenv
import os
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from src.rag_eval_project.generator import generate  
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL = os.getenv("JUDGE_MODEL_NAME")
THRESHOLD = 0.7


def run():
    # 1. LOAD the faithfulness golden set (query + ideal_context)
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE GENERATOR on the GOLDEN context (isolation), build one test case each
    test_cases = []
    for g in goldens:
        context = g["ideal_context"]              # known-good context (list of chunk strings)
        answer = generate(g["query"], context)    # RUN the generator -> actual_output

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,             # the generated answer we're judging
                retrieval_context=context,        # faithfulness checks the answer against THIS
                # no expected_output --- faithfulness never reads it
            )
        )

    # 3. THE METRICS --- decompose actual_output into claims, attribute each to context
    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,   # prints WHY each score --- shows which claims were unsupported
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
    ]

    # 4. EVALUATE --- runs the metrics on every case, prints a report
    result = evaluate(test_cases=test_cases, metrics=metrics)
    return summarize_by_metric(result)


if __name__ == "__main__":
    print_summary("generator", run())