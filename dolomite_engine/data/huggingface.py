from datasets import load_dataset
import json
from transformers import AutoTokenizer
from typing import List
import os

from ..enums import DatasetKeys, DatasetSplit, DatasetType, LossMask, Mode
from ..utils import run_local_rank_n_first
from ..utils import log_rank_0, logging
from .base import BaseDataset


class HuggingFaceDataset(BaseDataset):
    """A dataset class to load any HuggingFace dataset, expects a tuple of input and output keys"""

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
        type: DatasetType,
        loss_mask: LossMask,
        num_virtual_tokens: int = 0,
    ) -> None:
        super().__init__(
            class_args=class_args,
            split=split,
            mode=mode,
            tokenizer=tokenizer,
            is_encoder_decoder=is_encoder_decoder,
            data_name=data_name,
            input_format=input_format,
            output_format=output_format,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            loss_mask=loss_mask,
            num_virtual_tokens=num_virtual_tokens,
        )

        self.type = type
        self.examples = self.prepare_examples()

    def prepare_examples(self) -> List[dict]:
        assert "data_path" in self.class_args, "`data_path` is not specified"

        data_path: str = self.class_args.pop("data_path")
        if self.type == DatasetType.singleturn:
            input_key: str = self.class_args.pop("input_key", DatasetKeys.input.value)
            output_key: str = self.class_args.pop("output_key", DatasetKeys.output.value)
        elif self.type == DatasetType.multiturn:
            conversation_key: str = self.class_args.pop("conversation_key", DatasetKeys.conversation.value)
            conversation_role_key: str = self.class_args.pop("conversation_key", DatasetKeys.role.value)
            conversation_content_key: str = self.class_args.pop("conversation_key", DatasetKeys.content.value)

        split = "validation" if self.split == DatasetSplit.val else self.split.value
        dataset = load_dataset(data_path, **self.class_args)[split]

        with run_local_rank_n_first():
            if self.type == DatasetType.singleturn:
                func = lambda x: self.get_singleturn_ids(
                    x[input_key],
                    x[output_key] if self.mode == Mode.training else None
                )
            elif self.type == DatasetType.multiturn:
                func = lambda x: \
                    self.get_multiturn_ids([
                        # Huggingface standardized keys for apply_chat_template
                        {
                            "role": message[conversation_role_key], 
                            "content": message[conversation_content_key]
                        } for message in x[conversation_key]]
                        if conversation_role_key != "role" or conversation_content_key != "content" else 
                        x[conversation_key],
                        tools=[json.loads(tool) for tool in x["tools"]] if "tools" in dataset.column_names and x["tools"] is not None else None,
                    )

            dataset = dataset.map(
                func,
                num_proc=int(os.getenv("MAX_JOBS", 4)),
                remove_columns=[name for name in dataset.column_names if name not in {"labels", "input_ids", "uuid"}],
                desc=f"Tokenizing examples [{self.data_name}]",
            )

        log_rank_0(logging.INFO, f"[{self.data_name} example[0]]\n{self.tokenizer.decode(dataset[1]['input_ids'])}")

        return dataset
