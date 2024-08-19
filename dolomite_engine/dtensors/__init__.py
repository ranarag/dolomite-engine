from torch.distributed._tensor.api import DTensor


class DolomiteDTensor(DTensor):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()
