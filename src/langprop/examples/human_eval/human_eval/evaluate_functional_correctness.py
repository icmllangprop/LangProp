import fire
import sys

from langprop.examples.human_eval.human_eval.data import HUMAN_EVAL
from langprop.examples.human_eval.human_eval.evaluation import evaluate_functional_correctness


def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
    solution_complete: bool = False
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file, solution_complete)
    print(results)


def main():
    fire.Fire(entry_point)


sys.exit(main())