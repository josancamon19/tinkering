import statistics
import chz
import tinker
import datasets
from dotenv import load_dotenv
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinkering._1_gsm8k_manual import _run_eval
from enum import Enum

load_dotenv()


# Okay, so something interesting to point out here is that there are two types of things you can save in Tinker:
# 1. On every iteration you save the sampling parameters which basically means you're saving the lower matrices that you can use to sample from the model that's needed for inference etc
# 2. You can also save your whole checkpoint right that includes optimizers, statepaths, everything basically but in this case it would need it's like three changes heavy and it's not needed for sampling, like you just need the lower matrices


class Model(Enum):
    QWEN3_8B_BASE = (
        "tinker://4bd7a989-7c2a-58f9-9206-80757af9083e:train:0/sampler_weights/000027"
    )
    QWEN3_4B_INSTRUCT = (
        "tinker://a7688cff-a663-559f-8e52-118c83c314ec:train:0/sampler_weights/000025"
    )


@chz.chz
class Config:
    model: Model = Model.QWEN3_8B_BASE
    max_tokens: int = 256


def main(config: Config):
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        model_path=config.model.value
    )

    model_name = (
        "Qwen/Qwen3-8B-Base"
        if config.model == Model.QWEN3_8B_BASE
        else "Qwen/Qwen3-4B-Instruct-2507"
    )
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, get_tokenizer(model_name))
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens, stop=renderer.get_stop_sequences()
    )

    test_dataset = datasets.load_dataset("openai/gsm8k", "main")["test"]
    # test_dataset = test_dataset.select(range(100))
    accuracy, rewards = _run_eval(
        config, renderer, test_dataset, sampling_client, sampling_params, 0, None
    )
    print(f"Accuracy: {accuracy}")
    print(f"Rewards: {rewards}")
    print(f"Mean Reward: {sum(rewards) / len(rewards)}")
    print(f"Std Reward: {statistics.stdev(rewards)}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
