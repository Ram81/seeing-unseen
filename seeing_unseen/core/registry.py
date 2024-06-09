#!/usr/bin/env python3

r"""Registry is central source of truth.

Taken from Pythia, it is inspired from Redux's concept of global store.
Registry maintains mappings of various information to unique keys. Special
functions in registry can be used as decorators to register different kind of
classes.

Import the global registry object using

.. code:: py

    from habitat.core.registry import registry

Various decorators for registry different kind of classes with unique keys

-   Register a task: ``@registry.register_task``
-   Register a task action: ``@registry.register_task_action``
-   Register a simulator: ``@registry.register_simulator``
-   Register a sensor: ``@registry.register_sensor``
-   Register a measure: ``@registry.register_measure``
-   Register a dataset: ``@registry.register_dataset``
-   Register a environment: ``@registry.register_env``
"""

import collections
from typing import Any, Callable, DefaultDict, Dict, Optional, Type

from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset

from seeing_unseen.core.base import BaseTrainer, BaseTransform


class Singleton(type):
    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]


class Registry(metaclass=Singleton):
    mapping: DefaultDict[str, Any] = collections.defaultdict(dict)

    @classmethod
    def _register_impl(
        cls,
        _type: str,
        to_register: Optional[Any],
        name: Optional[str],
        assert_type: Optional[Type] = None,
    ) -> Callable:
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(
                    to_register, assert_type
                )
            register_name = to_register.__name__ if name is None else name

            cls.mapping[_type][register_name] = to_register
            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def register_dataset(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a dataset to registry with key :p:`name`

        :param name: Key with which the dataset will be registered.
            If :py:`None` will use the name of the class
        """

        return cls._register_impl(
            "dataset", to_register, name, assert_type=Dataset
        )

    @classmethod
    def register_affordance_model(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        r"""Register a affordance model to registry with key :p:`name`

        :param name: Key with which the affordance model will be registered.
            If :py:`None` will use the name of the class
        """

        return cls._register_impl(
            "affordance_model", to_register, name, assert_type=Module
        )

    @classmethod
    def register_loss_fn(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a loss function to registry with key :p:`name`

        :param name: Key with which the loss fn will be registered.
            If :py:`None` will use the name of the class
        """

        return cls._register_impl(
            "loss_fn", to_register, name, assert_type=Module
        )

    @classmethod
    def register_trainer(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a Trainer to registry with key :p:`name`

        :param name: Key with which the trainer will be registered.
            If :py:`None` will use the name of the class
        """

        return cls._register_impl(
            "trainer", to_register, name, assert_type=BaseTrainer
        )

    @classmethod
    def register_transforms(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        r"""Register a Transform to registry with key :p:`name`

        :param name: Key with which the trainer will be registered.
            If :py:`None` will use the name of the class
        """

        return cls._register_impl(
            "transforms", to_register, name, assert_type=BaseTransform
        )

    @classmethod
    def _get_impl(cls, _type: str, name: str) -> Type:
        return cls.mapping[_type].get(name, None)

    @classmethod
    def get_dataset(cls, name: str) -> Type[Dataset]:
        return cls._get_impl("dataset", name)

    @classmethod
    def get_affordance_model(cls, name: str) -> Type[Module]:
        return cls._get_impl("affordance_model", name)

    @classmethod
    def get_loss_fn(cls, name: str) -> Type[Module]:
        return cls._get_impl("loss_fn", name)

    @classmethod
    def get_trainer(cls, name: str) -> Type[BaseTrainer]:
        return cls._get_impl("trainer", name)

    @classmethod
    def get_transforms(cls, name: str) -> Type[BaseTransform]:
        return cls._get_impl("transforms", name)


registry = Registry()
