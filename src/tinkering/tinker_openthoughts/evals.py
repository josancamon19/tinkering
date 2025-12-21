from datasets import Dataset

import tinker
from tinker_cookbook.eval.evaluators import TrainingClientEvaluator
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.renderers import Renderer, TrainOnWhat

from tinkering.tinker_openthoughts.common import openthoughts_row_to_datum


class NLLEvaluator(TrainingClientEvaluator):
    def __init__(self, data: list[tinker.Datum], name: str = "test"):
        self.name = name
        self.data = data

    async def __call__(
        self, training_client: tinker.TrainingClient
    ) -> dict[str, float]:
        future = await training_client.forward_async(self.data, loss_fn="cross_entropy")
        result = await future.result_async()
        logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
        weights = [datum.loss_fn_inputs["weights"] for datum in self.data]
        nll = compute_mean_nll(logprobs, weights)
        key = f"{self.name}/nll"
        return {key: nll}

    @classmethod
    def from_split(
        cls,
        split: Dataset,
        renderer: Renderer,
        max_tokens: int,
        name: str = "test",
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> "NLLEvaluator":
        """Create an NLLEvaluator from a dataset split (e.g., split["test"])."""
        data = [
            openthoughts_row_to_datum(
                row, renderer, max_length=max_tokens, train_on_what=train_on_what
            )
            for row in split
        ]
        return cls(data, name=name)
