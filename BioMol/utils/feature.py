from typing import Any, overload
import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
import enum
import torch
import re


@enum.unique
class FeatureLevel(enum.Enum):
    ATOM = "atom"
    CHEMCOMP = "chem_comp"
    ENTITY = "entity"
    CHAIN = "chain"
    UNIT = "asymmetric_unit"
    FULL = "full"


class Help:
    def __init__(self, help: str | Sequence[str]):
        self.help = help

    def __call__(self):
        return self.help


class Feature(ABC):
    @abstractmethod
    def feature(self) -> Any:
        pass

    @abstractmethod
    def help(self) -> str | Sequence[str]:
        pass


class FeatureMap(ABC):
    @abstractmethod
    def feature(self, key: str | Sequence[str]) -> Feature | Sequence[Feature]:
        pass

    @abstractmethod
    @overload
    def help(self) -> str | Sequence[str]:
        pass

    @abstractmethod
    @overload
    def help(self, key: str | Sequence[str]) -> str | Sequence[str]:
        pass


@dataclasses.dataclass(frozen=True, slots=True)
class Feature0D(Feature):
    key: str
    value: Any
    level: FeatureLevel
    additional_info: Any

    def feature(self) -> str | int | float | bool | Sequence[str] | None:
        return self.value

    def help(self) -> str | Sequence[str]:
        return self.additional_info  # TODO : parse additional_info


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureMap0D(FeatureMap):
    feature_map: Mapping[str, Feature0D]

    def keys(self) -> list[str]:
        return list(self.feature_map.keys())

    def feature(
        self, key: str = None
    ) -> Mapping[str, Feature0D] | Feature0D:
        # Concrete implementation for feature
        if key is None:
            return self.feature_map
        return self.feature_map[key]

    def help(self, key: str = None) -> str | Sequence[str]:
        # Concrete implementation for help
        if key is None:
            help_list = []
            for items in self.feature_map.values():
                help = items.help()
                if isinstance(help, str):
                    help_list.append(help)
                elif isinstance(help, Sequence):
                    help_list.extend(help)
                else:
                    raise ValueError(
                        "help should be a string or a list of strings (TODO)"
                    )
            return help_list
        return self.feature_map[key].help()

    def __contains__(self, key):
        return key in self.feature_map

    def __repr__(self):
        output = "FeatureMap0D(\n"
        for value in self.feature_map.values():
            output += f"  {value}\n"
        output += ")"
        return output

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.feature_map[key]
        else:
            raise ValueError("Key should be a string for 0D feature map")


def calculate_token_length(s):
    """
    Calculate the length of a string where substrings enclosed in parentheses (~) are treated as single tokens.

    Args:
        s (str): Input string containing tokens like "(IVA)VV(LTA)".

    Returns:
        int: The calculated length.
    """  # noqa: E501
    # Use regex to match either parenthesized tokens or individual characters not in parentheses  # noqa: E501
    tokens = re.findall(r"\([^)]*\)|.", s)
    return len(tokens)


@dataclasses.dataclass(frozen=True, slots=True)
class Feature1D(Feature):
    key: str
    value: Sequence[str] | torch.Tensor
    mask: Sequence[bool] | torch.Tensor | None
    level: FeatureLevel
    additional_info: Any

    def feature(self) -> Sequence[str] | torch.Tensor | None:
        return self.value, self.mask

    def __getitem__(self, idx) -> str | int | float | bool | torch.Tensor:
        return self.value[idx], self.mask[idx]

    def __len__(self) -> int:
        if isinstance(self.value, str):
            return calculate_token_length(self.value)
        else:
            return len(self.value)

    def help(self) -> str | Sequence[str]:
        return self.additional_info  # TODO : parse additional_info


@dataclasses.dataclass(frozen=True)
class FeatureMap1D(FeatureMap):
    feature_map: Mapping[str, Feature1D]

    def keys(self) -> list[str]:
        return list(self.feature_map.keys())

    def feature(
        self, key: str = None
    ) -> Mapping[str, Feature1D] | Feature1D:
        # Concrete implementation for feature
        if key is None:
            return self.feature_map
        return self.feature_map[key]

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.feature_map[key]
        else:
            return {k: v[key] for k, v in self.feature_map.items()}

    def __contains__(self, key):
        return key in self.feature_map

    def help(self) -> Sequence[str]:
        help_list = []
        for items in self.feature_map.values():
            help = items.help()
            if isinstance(help, str):
                help_list.append(help)
            elif isinstance(help, Sequence[str]):
                help_list.extend(help)
            else:
                raise ValueError("help should be a string or a list of strings (TODO)")

        return help_list

    def __repr__(self):
        output = "FeatureMap1D(\n"
        for value in self.feature_map.values():
            output += f"  {value}\n"
        output += ")"
        return output


@dataclasses.dataclass(frozen=True)
class FeaturePair(Feature):
    key: str
    value: Mapping[tuple[int, int] | tuple[str, str], Any]
    level: FeatureLevel
    unidirectional: bool
    additional_info: Any

    def feature(self) -> Mapping[tuple[int, int] | tuple[str, str], Any]:
        return self.value

    def __getitem__(
        self, idx: tuple[int, int] | tuple[str, str]
    ) -> str | int | float | bool | torch.Tensor:
        return self.value[idx]

    def help(self) -> str | Sequence[str]:
        return self.additional_info  # TODO : parse additional_info


@dataclasses.dataclass(frozen=True)
class FeatureMapPair(FeatureMap):
    feature_map: Mapping[str, FeaturePair]

    def keys(self) -> list[str]:
        return list(self.feature_map.keys())

    def feature(
        self, key: str = None
    ) -> Mapping[str, FeaturePair] | FeaturePair:
        # Concrete implementation for feature
        if key is None:
            return self.feature_map
        return self.feature_map[key]

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.feature_map[key]
        else:
            return {k: v[key] for k, v in self.feature_map.items()}

    def __contains__(self, key):
        return key in self.feature_map

    def help(self) -> Sequence[str]:
        help_list = []
        for items in self.feature_map.values():
            help = items.help()
            if isinstance(help, str):
                help_list.append(help)
            elif isinstance(help, Sequence[str]):
                help_list.extend(help)
            else:
                raise ValueError("help should be a string or a list of strings (TODO)")

        return help_list

    def __repr__(self):
        output = "FeatureMapPair(\n"
        for value in self.feature_map.values():
            output += f"  {value}\n"
        output += ")"
        return output


class FeatureMapContainer:
    def __init__(
        self,
        feature_map_0D: FeatureMap0D | None,
        feature_map_1D: FeatureMap1D | None,
        feature_map_pair: FeatureMapPair | None,
    ):
        self.feature_map_0D = feature_map_0D if feature_map_0D is not None else {}
        self.feature_map_1D = feature_map_1D if feature_map_1D is not None else {}
        self.feature_map_pair = feature_map_pair if feature_map_pair is not None else {}
        self._check_duplicate_keys()
        self.key_to_map = {
            **self.feature_map_0D,
            **self.feature_map_1D,
            **self.feature_map_pair,
        }

        self._check_length()

    def _check_duplicate_keys(self):
        keys_0D = self.feature_map_0D.keys() if self.feature_map_0D is not None else []
        keys_1D = self.feature_map_1D.keys() if self.feature_map_1D is not None else []
        keys_pair = (
            self.feature_map_pair.keys() if self.feature_map_pair is not None else []
        )
        total_keys = list(keys_0D) + list(keys_1D) + list(keys_pair)
        if len(set(total_keys)) != len(total_keys):
            raise ValueError("Duplicate keys found in feature maps")

    def _check_length(self):
        length_dict = {}
        if self.feature_map_1D is None or self.feature_map_1D == {}:
            return
        for key in self.feature_map_1D.keys():
            length_dict[key] = len(self.feature_map_1D[key])

        all_lengths = list(length_dict.values())
        if len(set(all_lengths)) != 1:
            raise ValueError("Length of all 1D features should be same")

    def __getitem__(
        self, idx: str | int | slice | tuple[int, int]
    ) -> Feature0D | Feature1D | FeaturePair:
        if isinstance(idx, str):
            return self.key_to_map[idx]
        elif isinstance(idx, int) or isinstance(idx, slice):
            return self.feature_map_1D[idx]
        elif isinstance(idx, tuple[int, int]):
            return self.feature_map_pair[idx]

    def get_0d_features(self) -> FeatureMap0D:
        return self.feature_map_0D

    def get_1d_features(self) -> FeatureMap1D:
        return self.feature_map_1D

    def get_pair_features(self) -> FeatureMapPair:
        return self.feature_map_pair
