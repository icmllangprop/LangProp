import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from langprop.examples.human_eval.human_eval.data import write_jsonl, read_problems, HUMAN_EVAL
from langprop.examples.human_eval.human_eval.execution import check_correctness
from langprop.module import LPModule, RunConfig
from langprop.trainer import LPTrainer


class HumanEvalModule(LPModule):
    def forward(self, script, *args, problem=None, **kwargs):
        return check_correctness(problem, script, timeout=20, solution_complete=True)


class HumanEvalTrainer(LPTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_scripts = []

    def preprocess(self, problem):
        return (), {"problem": problem}, problem

    def score(self, result, problem) -> float:
        return float(result["passed"])

    def _setup(self, *args, **kwargs):
        super()._setup(*args, **kwargs)
        self.best_scripts.append(str(self.module.script_records[0].script))

    def fit_batch(self, *args, **kwargs):
        super().fit_batch(*args, **kwargs)
        self.best_scripts.append(str(self.module.script_records[0].script))


def train_human_eval_model(problem, epochs=0, timestamp="", n_responses=2, n_top_choices=3):
    data_loader = [[problem]]
    model = HumanEvalModule.from_template(name="solve_human_eval", root=Path(__file__).parent, return_function=True)
    run_name = timestamp + "_human_eval_" + problem["task_id"]
    trainer = HumanEvalTrainer(model, RunConfig(run_name=run_name, n_responses=n_responses, n_top_choices=n_top_choices))
    trainer.fit(data_loader, epochs=epochs)
    return trainer.best_scripts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default=HUMAN_EVAL)
    parser.add_argument("--n-samples", "-ns", type=int, default=1)
    parser.add_argument("--n-responses", "-nr", type=int, default=2)
    parser.add_argument("--n-top-choices", "-nc", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=0)
    args = parser.parse_args()

    problems = read_problems(args.problem)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    samples_dict = defaultdict(list)

    for _ in range(args.n_samples):
        for task_id in problems:
            scripts = train_human_eval_model(problems[task_id], args.epochs, timestamp, n_responses=args.n_responses, n_top_choices=args.n_top_choices)
            for k in range(len(scripts)):
                samples_dict[k].append(dict(task_id=task_id, completion=scripts[k]))

    for k, samples in samples_dict.items():
        write_jsonl(f"HumanEval_{timestamp}_samples_pass_{k}.jsonl", samples)
