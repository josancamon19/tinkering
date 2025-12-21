from tinker_cookbook import renderers
from tinker_cookbook.renderers import TrainOnWhat, Message
from tinker_cookbook.supervised.common import datum_from_model_input_weights
import tinker


def openthoughts_row_to_datum(
    row: dict,
    renderer: renderers.Renderer,
    max_length: int | None = None,
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
) -> tinker.Datum:
    """
    Convert an OpenThoughts dataset row to a tinker.Datum.

    OpenThoughts format:
    {
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "<think>...</think>..."}
        ]
    }

    This converts to chat Message format and uses the renderer to build
    a supervised example with proper loss masking (only train on assistant response).
    """
    conversations = row["conversations"]
    messages: list[Message] = []

    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append(Message(role=role, content=turn["value"]))

    # Build the supervised example using the renderer
    # This handles tokenization, masking, and proper formatting for the model
    # weights it's such a bad name for a mask.!!!
    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=train_on_what
    )

    return datum_from_model_input_weights(model_input, weights, max_length)
