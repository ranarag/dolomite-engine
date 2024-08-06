from typing import Iterable

import torch

from ..enums import LossMask, Mode
from ..utils import print_ranks_all


def collate_fn(
    batch: list[dict],
    mode: Mode,
    eos_token_id: int,
    is_encoder_decoder: bool,
    use_padding_free_transformer: bool,
    labels_mask_value: int = -100,
) -> dict:
    """prepares the batch with padding to pass into the forward function of the HuggingFace model

    Args:
        batch (list[dict]): input tokens and output tokens. Output tokens are optional when running generation but required for training.

    Returns:
        dict: dict containing input_ids, attention_mask and labels if outputs is specified
    """

    input_ids = [i["input_ids"] for i in batch]
    labels = [i["labels"] for i in batch] if mode == Mode.training else None

    # print_ranks_all([i["uuid"] for i in batch])

    if is_encoder_decoder:
        if use_padding_free_transformer:
            raise NotImplementedError("padding free transformer only supports decoder only models")

        input_max_length = max(list(map(len, input_ids)))

        input_ids = [[eos_token_id] * (input_max_length - len(array)) + array for array in input_ids]
        attention_mask = [[0] * (input_max_length - len(array)) + [1] * len(array) for array in input_ids]

        if labels is not None:
            output_max_length = max(list(map(len, labels)))
            # right padding for labels
            labels = [array + [labels_mask_value] * (output_max_length - len(array)) for array in labels]
    else:
        if not use_padding_free_transformer:
            max_length = max(list(map(len, input_ids)))

            input_ids = [[eos_token_id] * (max_length - len(array)) + array for array in input_ids]
            attention_mask = [[0] * (max_length - len(array)) + [1] * len(array) for array in input_ids]
            labels = [[labels_mask_value] * (max_length - len(array)) + array for array in labels]

    if not use_padding_free_transformer:
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        if labels is not None:
            labels = torch.tensor(labels)

    result = {"input_ids": input_ids}

    if not use_padding_free_transformer:
        result["attention_mask"] = attention_mask

    if mode == Mode.training:
        result["labels"] = labels

    return result


def infinite_iterator(x: Iterable | None) -> Iterable:
    """converts and iterable into a non-ending infinite iterable, will return None if input is None

    Args:
        x (Iterable): the iterable to convert

    Returns:
        Iterable: the converted iterable

    Yields:
        Iterator[Iterable]: an element from the original iterator
    """

    if x is None:
        return None

    while True:
        for i in x:
            yield i


def get_next_batch(x: Iterable | None) -> dict:
    """get next batch

    Args:
        x (Iterable): dataloader

    Returns:
        dict: batch
    """

    # train_dataloader is always None on TP ranks other than 0
    if x is None:
        return None

    return next(x)
