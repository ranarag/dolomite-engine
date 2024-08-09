import copy
import random
from faker import Faker
import torch
from transformers import AutoTokenizer
from typing import List, Dict

from ..defaults import INPUT_FORMAT, OUTPUT_FORMAT
from ..enums import DatasetSplit, LossMask, Mode
from ..utils import logging, log_rank_0


class BaseDataset(torch.utils.data.Dataset):
    """BaseDataset class to be implemented by all the datasets"""

    def __init__(
        self,
        class_args: dict,
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
        data_name: str,
        input_format: str,
        output_format: str,
        max_input_tokens: int,
        max_output_tokens: int,
        max_total_tokens: int,
        loss_mask: LossMask,
        num_virtual_tokens: int = 0,
    ) -> None:
        super().__init__()

        self.split = split
        self.mode = mode

        self.class_args = class_args

        self.tokenizer = tokenizer
        self.is_encoder_decoder = is_encoder_decoder

        # used for prompt tuning
        self.num_virtual_tokens = num_virtual_tokens

        self.data_name = data_name
        self.input_format = input_format
        self.output_format = output_format
        self.loss_mask = loss_mask

        # if format is __input__ or __output__ formatting is a no-op
        self.do_format_input = self.input_format != INPUT_FORMAT
        self.do_format_output = self.output_format != OUTPUT_FORMAT

        # length to use for trimming (excludes eos)
        self.max_input_tokens = get_max_input_length(
            max_input_tokens, self.num_virtual_tokens, self.is_encoder_decoder
        )
        self.max_output_tokens = get_max_output_length(
            max_output_tokens, self.num_virtual_tokens, self.is_encoder_decoder
        )
        self.max_total_tokens = max_total_tokens

        self.examples = []

    def _construct_input_from_format(self, input: str) -> str:
        """construct input with the specified input_format

        Args:
            input (str): input text

        Returns:
            str: formatted text
        """

        if self.do_format_input:
            return self.input_format.replace(INPUT_FORMAT, input, 1)
        return input

    def _construct_output_from_format(self, output: str) -> str:
        """construct output with the specified output_format

        Args:
            output (str): output text

        Returns:
            str: formatted text
        """

        if self.do_format_output:
            return self.output_format.replace(OUTPUT_FORMAT, output, 1)
        return output

    def get_singleturn_ids(self, input: str, output: str) -> Dict[str, List[int]]:
        input_formatted = self._construct_input_from_format(input)
        output_formatted = self._construct_output_from_format(output)

        if self.is_encoder_decoder:
            input_ids = self.tokenizer(input_formatted, add_special_tokens=False)["input_ids"]
            if self.max_input_tokens is not None:
                input_ids = input_ids[: self.max_input_tokens - 1]
            input_ids.append(self.tokenizer.eos_token_id)

            if self.mode == Mode.training:
                labels = self.tokenizer(output_formatted, add_special_tokens=False)["input_ids"]
                if self.max_output_tokens is not None:
                    labels = labels[: self.max_output_tokens - 1]
                labels.append(self.tokenizer.eos_token_id)
            else:
                labels = None
        else:
            if self.mode == Mode.training:
                if self.loss_mask in {LossMask.output_prompted, LossMask.no_mask_prompted}:
                    input_ids = self.tokenizer(
                        input_formatted + self.output_format.split(OUTPUT_FORMAT)[0] ,
                        add_special_tokens=False, 
                    )["input_ids"]

                    output_ids = self.tokenizer(
                        output + self.output_format.split(OUTPUT_FORMAT)[1] + self.tokenizer.eos_token,
                        add_special_tokens=False, 
                    )["input_ids"]

                    if self.loss_mask == LossMask.output_prompted:
                        labels = [*[-100] * len(input_ids), *output_ids]
                    else:
                        labels = [*input_ids, *output_ids]

                    input_ids.extend(output_ids)
                elif self.loss_mask in {LossMask.output, LossMask.no_mask}:
                    input_ids = self.tokenizer(
                        input_formatted,
                        add_special_tokens=False, 
                    )["input_ids"]

                    output_ids = self.tokenizer(
                        output_formatted + self.tokenizer.eos_token,
                        add_special_tokens=False, 
                    )["input_ids"]

                    if self.loss_mask == LossMask.output:
                        labels = [*[-100] * len(input_ids), *output_ids]
                    else:
                        labels = [*input_ids, *output_ids]
                    
                    input_ids.extend(output_ids)
            else:
                input_ids = self.tokenizer(
                    input_formatted, 
                    add_special_tokens=False, 
                )

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def get_multiturn_ids(self, conversation: List[Dict[str, str]], tools = None) -> Dict[str, List[int]]:
        # print(tools)
        use_system = False #random.random() < self.tokenizer.system_probability
        date_string = "None" #Faker().date_between(start_date='-1y', end_date='+1y').strftime("%B %d, %Y")

        assert (
            not self.is_encoder_decoder
        ), "Multi-turn conversations are not supported with encoder decoder models. Please reformat into a single turn"

        assert (
            (self.do_format_output or self.do_format_input) or self.tokenizer.chat_template is not None
        ), "Must specify either an input/output format or chat template to use multiturn samples"

        assert (
            self.max_input_tokens is None and self.max_output_tokens is None
        ), "Must use `max_total_tokens` for truncating multiturn datasets. Please unset `max_input_tokens` and `max_output_tokens`"

        if self.do_format_output or self.do_format_input:
            if self.tokenizer.chat_template is not None:
                log_rank_0(logging.INFO, f"Overriding tokenizer chat template with `{self.data_name}`'s format")
            pass
        elif self.tokenizer.chat_template is not None:
            input_ids = self.tokenizer.apply_chat_template(
                conversation, 
                tokenize=True, 
                add_generation_prompt=False, 
                return_tensors="pt",
                training=True,
                use_system=use_system,
                date_string=date_string,
                tools=tools,
            )

            labels = input_ids.clone()

            if self.loss_mask in {LossMask.output, LossMask.output_prompted}:
                for message_idx, message in enumerate(conversation):
                    if message["role"] != "assistant" and message["role"] != "assistant_tool_call":
                        if message_idx == 0:
                            message_start_idx = 0
                        else:
                            message_start_idx = self.tokenizer.apply_chat_template(
                                conversation[:message_idx], 
                                tokenize=True, 
                                add_generation_prompt=False, 
                                return_tensors="pt",
                                training=True,
                                use_system=use_system,
                                date_string=date_string,
                                tools=tools,
                            ).shape[1]

                        if message_idx < len(conversation) - 1 and conversation[message_idx + 1]["role"] == "assistant":
                            message_end_idx = self.tokenizer.apply_chat_template(
                                conversation[:message_idx+1], 
                                tokenize=True, 
                                add_generation_prompt=True, 
                                return_tensors="pt",
                                training=True,
                                use_system=use_system,
                                date_string=date_string,
                                tools=tools,
                            ).shape[1]
                        else:
                            message_end_idx = self.tokenizer.apply_chat_template(
                                conversation[:message_idx+1], 
                                tokenize=True, 
                                add_generation_prompt=False,
                                return_tensors="pt",
                                training=True,
                                use_system=use_system,
                                date_string=date_string,
                                tools=tools,
                            ).shape[1]

                        labels[:, message_start_idx:message_end_idx] = -100
        if self.max_total_tokens:
            return {
                "input_ids": input_ids.flatten()[:self.max_total_tokens],
                "labels": labels.flatten()[:self.max_total_tokens],
            }
        return {
            "input_ids": input_ids.flatten(),
            "labels": labels.flatten(),
        }

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        return

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]

    def __len__(self) -> int:
        return len(self.examples)


class BlendedDatasets(torch.utils.data.Dataset):
    """Concatenated list of datasets for training or inference"""

    def __init__(self, datasets: list[BaseDataset], split: DatasetSplit) -> None:
        super().__init__()

        self.split = split
        self.datasets = datasets

        self.num_examples = sum(self.get_num_examples_in_each_dataset())
        self.indexing_array = self._get_indexing_array()

    def get_num_datasets(self) -> int:
        """returns the number of datasets in the mixture

        Returns:
            int: number of datasets in the mixture
        """

        return len(self.datasets)

    def get_num_examples_in_each_dataset(self) -> list[int]:
        """returns the number of examples in each dataset component

        Returns:
            list[int]: the number of examples in each dataset component
        """

        return [len(dataset) for dataset in self.datasets]

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        return

    def _get_indexing_array(self) -> list[tuple[int]]:
        num_examples_in_each_dataset = self.get_num_examples_in_each_dataset()

        indexing_array = []
        for dataset_index, num_examples in enumerate(num_examples_in_each_dataset):
            for example_id in range(num_examples):
                indexing_array.append((dataset_index, example_id))

        return indexing_array

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> dict:
        dataset_index, example_index = self.indexing_array[index]
        example = self.datasets[dataset_index][example_index]
        return example

    def __repr__(self) -> str:
        x = f"number of datasets = {self.get_num_datasets()}\n"
        x += f"total examples in the entire dataset mixture = {len(self)}"

        for dataset in self.datasets:
            x += f"\nexamples in {dataset.__class__.__name__} ({dataset.data_name}) = {len(dataset)}"

        return x


def get_max_input_length(
    max_input_tokens_specified: int | None, num_virtual_tokens: int, is_encoder_decoder: bool
) -> int:
    """max input length for the model, depends on the training / inference type and whether the model is decoder-only or encoder-decoder

    Args:
        max_input_tokens_specified (int | None): maximum number of specified input tokens
        num_virtual_tokens (int): virtual tokens for prompt tuning
        is_encoder_decoder (bool): whether the model is decoder-only or encoder-decoder

    Returns:
        int: max input length
    """

    if max_input_tokens_specified is None:
        return None

    max_input_tokens = max_input_tokens_specified - num_virtual_tokens

    if is_encoder_decoder:
        max_input_tokens -= 1

    return max_input_tokens


def get_max_output_length(
    max_output_tokens_specified: int | None, num_virtual_tokens: int, is_encoder_decoder: bool
) -> int:
    """max output length for the model, depends on the training / inference type and whether the model is decoder-only or encoder-decoder

    Args:
        max_output_tokens_specified (int | None): maximum number of specified output tokens
        num_virtual_tokens (int): virtual tokens for prompt tuning
        is_encoder_decoder (bool): whether the model is decoder-only or encoder-decoder

    Returns:
        int: max output length
    """

    if max_output_tokens_specified is None:
        return None

    max_output_tokens = max_output_tokens_specified - 1

    if is_encoder_decoder:
        max_output_tokens -= num_virtual_tokens

    return max_output_tokens
