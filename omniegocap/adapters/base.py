"""
Adapters turn a device recording into the model input (omniegocap.io.sequence.EgoSequence).

The contract is EgoSequence, not the adapter: any code that fills an EgoSequence works. An adapter only packages
that conversion for scripts/prepare_input.py:

    @register
    class MyDevice(Adapter):
        name = 'my_device'

        @classmethod
        def add_arguments(cls, parser):
            parser.add_argument('--recording', type=str, required=True)

        def load(self, args):
            ...
            return EgoSequence(head=..., wrist=..., visible=...)

A module outside this package is picked up with `scripts/prepare_input.py --plugin my_package.my_device ...`.
"""

import argparse
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Type

from omniegocap.io.sequence import EgoSequence

ADAPTERS: Dict[str, Type['Adapter']] = {}


class Adapter(ABC):
    name: ClassVar[str]


    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """
        Declare the command line arguments of this adapter.

        Args:
            parser (argparse.ArgumentParser):
                Sub-parser of this adapter.
        """


    @abstractmethod
    def load(self, args: argparse.Namespace) -> EgoSequence:
        """
        Read the recording and return the model input.

        Args:
            args (argparse.Namespace):
                Parsed arguments.

        Returns:
            seq (EgoSequence):
                Head and wrist poses in a z-up world with the floor at z = 0, at 30 fps.
        """


def register(cls: Type[Adapter]) -> Type[Adapter]:
    """
    Class decorator that makes an adapter available to scripts/prepare_input.py.

    Args:
        cls (Type[Adapter]):
            Adapter class with a unique `name`.

    Returns:
        cls (Type[Adapter]):
            The same class.
    """

    if cls.name in ADAPTERS:
        raise ValueError(f'Adapter "{cls.name}" is already registered')
    ADAPTERS[cls.name] = cls

    return cls
