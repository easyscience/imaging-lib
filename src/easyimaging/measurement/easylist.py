#  SPDX-FileCopyrightText: 2026 EasyImaging contributors  <imaging@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2026 Contributors to the EasyImaging project <https://github.com/easyScience/EasyImaging>

from __future__ import annotations

from collections.abc import MutableSequence
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import overload



class EasyList(MutableSequence):
    """
    A collection class for NewBase/ModelBase objects.
    This provides list-like functionality while maintaining EasyScience features
    like serialization.
    """

    def __init__(
        self,
        allowed_type: Any = None,
    ):
        """
        Initialize the EasyList.

        :param args: Initial items to add to the collection
        :param unique_name: Optional unique name for the collection
        :param display_name: Optional display name for the collection
        """

        self._data = []
        self._allowed_type = allowed_type

    # MutableSequence abstract methods

    # Use @overload to provide precise type hints for different __getitem__ argument types
    @overload
    def __getitem__(self, idx: int) -> T: ...
    @overload
    def __getitem__(self, idx: slice) -> 'EasyList': ...
    @overload
    def __getitem__(self, idx: str) -> T: ...

    def __getitem__(self, idx: int | slice | str) -> 'EasyList':
        """
        Get an item by index, slice, or name.

        :param idx: Index, slice, or name of the item
        :return: The item or a new collection for slices
        """
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return self.__class__(*[self._data[i] for i in range(start, stop, step)])
        if isinstance(idx, str):
            # Search by name
            for item in self._data:
                if hasattr(item, 'name') and getattr(item, 'name') == idx:
                    return item  # type: ignore[return-value]
                if hasattr(item, 'unique_name') and item.unique_name == idx:
                    return item  # type: ignore[return-value]
            raise KeyError(f'No item with name "{idx}" found')
        return self._data[idx]  # type: ignore[return-value]

    @overload
    def __setitem__(self, idx: int, value: T) -> None: ...
    @overload
    def __setitem__(self, idx: slice, value: Iterable[T]) -> None: ...

    def __setitem__(self, idx: int | slice, value: T | Iterable[T]) -> None:
        """
        Set an item at an index.

        :param idx: Index to set
        :param value: New value
        """
        if isinstance(idx, slice):
            # Handle slice assignment
            values = list(value)  # type: ignore[arg-type]
            # Remove old items
            start, stop, step = idx.indices(len(self))
            for i in range(start, stop, step):
                self._remove_item(self._data[i])
            # Set new items
            self._data[idx] = values  # type: ignore[assignment]
            for v in values:
                self._global_object.map.add_edge(self, v)
                self._global_object.map.reset_type(v, 'created_internal')
        else:
            if not isinstance(value, NewBase):
                raise TypeError(f'Items must be NewBase objects, got {type(value)}')

            old_item = self._data[idx]
            self._remove_item(old_item)

            self._data[idx] = value  # type: ignore[assignment]
            self._global_object.map.add_edge(self, value)
            self._global_object.map.reset_type(value, 'created_internal')

    @overload
    def __delitem__(self, idx: int) -> None: ...
    @overload
    def __delitem__(self, idx: slice) -> None: ...
    @overload
    def __delitem__(self, idx: str) -> None: ...

    def __delitem__(self, idx: int | slice | str) -> None:
        """
        Delete an item by index, slice, or name.

        :param idx: Index, slice, or name of item to delete
        """
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            indices = list(range(start, stop, step))
            # Remove in reverse order to maintain indices
            for i in reversed(indices):
                item = self._data[i]
                self._remove_item(item)
                del self._data[i]
        elif isinstance(idx, str):
            for i, item in enumerate(self._data):
                if hasattr(item, 'name') and getattr(item, 'name') == idx:
                    idx = i
                    break
                if hasattr(item, 'unique_name') and item.unique_name == idx:
                    idx = i
                    break
            else:
                raise KeyError(f'No item with name "{idx}" found')

            item = self._data[idx]
            self._remove_item(item)
            del self._data[idx]
        else:
            item = self._data[idx]
            self._remove_item(item)
            del self._data[idx]

    def __len__(self) -> int:
        """Return the number of items in the collection."""
        return len(self._data)

    def insert(self, index: int, value: T) -> None:
        """
        Insert an item at an index.

        :param index: Index to insert at
        :param value: Item to insert
        """
        if not isinstance(value, NewBase):
            raise TypeError(f'Items must be NewBase objects, got {type(value)}')

        self._data.insert(index, value)  # type: ignore[arg-type]
        self._global_object.map.add_edge(self, value)
        self._global_object.map.reset_type(value, 'created_internal')

    # Additional utility methods

    @property
    def data(self) -> tuple:
        """Return the data as a tuple."""
        return tuple(self._data)

    def sort(self, mapping: Callable[[T], Any], reverse: bool = False) -> None:
        """
        Sort the collection according to the given mapping.

        :param mapping: Mapping function to sort by
        :param reverse: Whether to reverse the sort
        """
        self._data.sort(key=mapping, reverse=reverse)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} of length {len(self)}'

    def __iter__(self) -> Any:
        return iter(self._data)

    # Serialization support

    def _convert_to_dict(self, in_dict: dict, encoder: Any, skip: Optional[List[str]] = None, **kwargs: Any) -> dict:
        """Convert the collection to a dictionary for serialization."""
        if skip is None:
            skip = []
        d: dict = {}
        if hasattr(self, '_modify_dict'):
            d = self._modify_dict(skip=skip, **kwargs)  # type: ignore[attr-defined]
        in_dict['data'] = [encoder._convert_to_dict(item, skip=skip, **kwargs) for item in self._data]
        return {**in_dict, **d}

    def get_all_variables(self) -> List[Any]:
        """Get all variables from all items in the collection."""
        variables: List[Any] = []
        for item in self._data:
            if hasattr(item, 'get_all_variables'):
                variables.extend(item.get_all_variables())  # type: ignore[attr-defined]
        return variables