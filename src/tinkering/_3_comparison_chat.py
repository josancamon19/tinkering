"""
Double-window comparison chat CLI.
Sends the same prompt to a base model and a checkpoint, displays completions side-by-side.
"""

import asyncio
import logging
import os
import sys
from enum import Enum

import chz
import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from dotenv import load_dotenv
from tinkering._1_gsm8k_manual import _get_1_shot_prefix, _get_gsm8k_question_suffix

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)-4s %(message)s",
    level=logging.WARNING,
    datefmt="%Y-%m-%d %H:%M:%S",
)

console = Console()


class Checkpoint(Enum):
    QWEN3_8B_BASE = (
        "tinker://4bd7a989-7c2a-58f9-9206-80757af9083e:train:0/sampler_weights/000027"
    )
    QWEN3_4B_INSTRUCT = (
        "tinker://a7688cff-a663-559f-8e52-118c83c314ec:train:0/sampler_weights/000025"
    )


CHECKPOINT_TO_BASE = {
    Checkpoint.QWEN3_8B_BASE: "Qwen/Qwen3-8B-Base",
    Checkpoint.QWEN3_4B_INSTRUCT: "Qwen/Qwen3-4B-Instruct-2507",
}


@chz.chz
class Config:
    checkpoint: Checkpoint = Checkpoint.QWEN3_8B_BASE
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    gsm8k_mode: bool = False  # Start with GSM8K mode off


class DualChatSession:
    """Manages dual chat sessions for base model and checkpoint comparison.

    Stateless: each prompt is independent, no conversation history is kept.
    """

    def __init__(
        self,
        base_client: tinker.SamplingClient,
        checkpoint_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        max_tokens: int,
        temperature: float,
        top_p: float,
        gsm8k_mode: bool = False,
    ):
        self.base_client = base_client
        self.checkpoint_client = checkpoint_client
        self.renderer = renderer
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.gsm8k_mode = gsm8k_mode

    def toggle_gsm8k_mode(self) -> bool:
        """Toggle GSM8K mode and return new state."""
        self.gsm8k_mode = not self.gsm8k_mode
        return self.gsm8k_mode

    def _build_prompt_messages(self, user_input: str) -> list[renderers.Message]:
        """Build messages for a single prompt.

        In GSM8K mode, prepends 1-shot example and appends boxed suffix.
        """
        if self.gsm8k_mode:
            messages = [
                *_get_1_shot_prefix(),
                {
                    "role": "user",
                    "content": user_input + _get_gsm8k_question_suffix(),
                },
            ]
            return messages
        else:
            return [{"role": "user", "content": user_input}]

    async def _generate_single(
        self, client: tinker.SamplingClient, model_input: types.ModelInput
    ) -> str:
        sampling_params = types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.renderer.get_stop_sequences(),
        )

        response = await client.sample_async(
            prompt=model_input, num_samples=1, sampling_params=sampling_params
        )

        parsed_message, _ = self.renderer.parse_response(response.sequences[0].tokens)
        return renderers.ensure_text(parsed_message["content"])

    async def generate_responses(self, user_input: str) -> tuple[str, str]:
        """Generate responses from both base model and checkpoint in parallel."""
        messages = self._build_prompt_messages(user_input)
        model_input = self.renderer.build_generation_prompt(messages)

        base_task = asyncio.create_task(
            self._generate_single(self.base_client, model_input)
        )
        checkpoint_task = asyncio.create_task(
            self._generate_single(self.checkpoint_client, model_input)
        )

        try:
            base_response, checkpoint_response = await asyncio.gather(
                base_task, checkpoint_task
            )
            return base_response, checkpoint_response
        except Exception as e:
            logger.error(f"Error generating responses: {e}")
            return f"Error: {e}", f"Error: {e}"


def create_status_bar(gsm8k_mode: bool, checkpoint_name: str) -> Table:
    """Create a compact status bar showing current settings."""
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left")
    table.add_column(justify="right")

    gsm8k_indicator = "[bold green]● ON[/]" if gsm8k_mode else "[dim]○ OFF[/]"

    table.add_row(
        f"[dim]GSM8K Mode:[/] {gsm8k_indicator}",
        f"[dim]Checkpoint:[/] [cyan]{checkpoint_name}[/]",
    )
    return table


def create_comparison_display(
    base_response: str,
    checkpoint_response: str,
    base_label: str,
    checkpoint_label: str,
) -> Group:
    """Create stacked panels for comparison with better visual hierarchy."""
    base_panel = Panel(
        Text(base_response, style="white"),
        title=f"[bold cyan]◀ {base_label}[/]",
        subtitle="[dim cyan]base model[/]",
        border_style="cyan",
        padding=(1, 2),
    )
    checkpoint_panel = Panel(
        Text(checkpoint_response, style="white"),
        title=f"[bold green]◀ {checkpoint_label}[/]",
        subtitle="[dim green]fine-tuned[/]",
        border_style="green",
        padding=(1, 2),
    )
    return Group(base_panel, checkpoint_panel)


def print_help():
    """Print available commands."""
    help_table = Table.grid(padding=(0, 2))
    help_table.add_column(style="cyan", justify="right")
    help_table.add_column(style="dim")

    help_table.add_row("g", "Toggle GSM8K mode (1-shot prefix + boxed suffix)")
    help_table.add_row("h", "Show this help")
    help_table.add_row("q", "Quit")

    console.print(Panel(help_table, title="[bold]Commands[/]", border_style="dim"))


async def main(config: Config):
    """Main comparison chat loop."""
    base_model = CHECKPOINT_TO_BASE[config.checkpoint]
    short_base = base_model.split("/")[-1]
    short_checkpoint = config.checkpoint.name

    # Header
    console.print()
    header = Table.grid(padding=1)
    header.add_column(justify="center")
    header.add_row("[bold magenta]🔬 Model Comparison Chat[/]")
    header.add_row(f"[cyan]{short_base}[/] vs [green]{short_checkpoint}[/]")
    console.print(Panel(header, border_style="magenta", padding=(0, 2)))

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        service_client = tinker.ServiceClient()

        with console.status("[dim]Initializing clients...[/]", spinner="dots"):
            base_client = service_client.create_sampling_client(base_model=base_model)
            checkpoint_client = service_client.create_sampling_client(
                model_path=config.checkpoint.value
            )
            tokenizer = get_tokenizer(base_model)
            renderer = renderers.get_renderer(
                get_recommended_renderer_name(base_model), tokenizer
            )

        chat_session = DualChatSession(
            base_client=base_client,
            checkpoint_client=checkpoint_client,
            renderer=renderer,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            gsm8k_mode=config.gsm8k_mode,
        )

        console.print()
        console.print("[green]✓[/] Ready!")
        console.print()
        print_help()
        console.print()

        while True:
            try:
                # Show status bar
                console.print(
                    create_status_bar(chat_session.gsm8k_mode, short_checkpoint)
                )
                user_input = console.input("[bold yellow]▶[/] ").strip()

                if not user_input:
                    continue

                # Commands
                if user_input.lower() in ["quit", "exit", "q"]:
                    console.print("\n[bold]👋 Goodbye![/]")
                    break

                if user_input.lower() == "g":
                    new_state = chat_session.toggle_gsm8k_mode()
                    state_str = "[green]ON[/]" if new_state else "[dim]OFF[/]"
                    console.print(f"\n[bold]GSM8K mode:[/] {state_str}\n")
                    if new_state:
                        console.print(
                            "[dim]Prompts will include 1-shot example + \\boxed{} suffix[/]\n"
                        )
                    continue

                if user_input.lower() == "h":
                    console.print()
                    print_help()
                    console.print()
                    continue

                console.print()
                with console.status("[bold cyan]Generating...[/]", spinner="dots"):
                    (
                        base_response,
                        checkpoint_response,
                    ) = await chat_session.generate_responses(user_input)

                console.print(
                    create_comparison_display(
                        base_response,
                        checkpoint_response,
                        short_base,
                        short_checkpoint,
                    )
                )
                console.print()

            except KeyboardInterrupt:
                console.print("\n\n[bold]👋 Interrupted. Goodbye![/]")
                break
            except EOFError:
                console.print("\n\n[bold]👋 Goodbye![/]")
                break
            except Exception as e:
                console.print(f"\n[bold red]❌ Error:[/] {e}")
                logger.exception("Unexpected error in chat loop")

    except Exception as e:
        console.print(f"\n[bold red]❌ Failed to initialize:[/] {e}")
        logger.exception("Failed to initialize")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
