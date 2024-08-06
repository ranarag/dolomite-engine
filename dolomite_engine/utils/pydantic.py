from copy import deepcopy
from enum import Enum

from pydantic import BaseModel, ConfigDict


class BaseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def to_dict(self) -> dict:
        def _serialize(value):
            if isinstance(value, Enum):
                return value.value
            elif isinstance(value, type):
                return value.__name__
            else:
                return value
        
        copied = deepcopy(self)

        for key, value in copied:
            if isinstance(value, BaseArgs):
                result = value.to_dict()
            elif isinstance(value, list):
                result = []
                for v in value:
                    if isinstance(v, BaseArgs):
                        result.append(v.to_dict())
                    else:
                        result.append(_serialize(v))
            else:
                result = _serialize(value)

            setattr(copied, key, result)

        return vars(copied)
