# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.
"""
Central Tag Management class
"""
from typing import Any, Dict, List, Union
import logging
import pandas as pd

from .dependencies import DependencyGraph
from .validation import validate_td, TagDictError


logger = logging.getLogger(__name__)


class TagDict:
    """
    Class to hold a data dictionary. Uses a dataframe underneath and takes care of
    QA and convenience methods.
    """

    def __init__(self, data: pd.DataFrame, validate: bool = True):
        """
        Default constructor. Creates new TagDict object from pandas dataframe.

        Args:
            data: input dataframe
            validate: whether to validate the input dataframe. validate=False can
             lead to a dysfunctional TagDict but may be useful for testing
        """
        self._validate = validate
        self._data = validate_td(data) if self._validate else data

        self._update_dependency_graph()

    def _update_dependency_graph(self):
        """
        Update dependency graph to reflect what is currently in the tag dict
        """
        graph = DependencyGraph()
        if "on_off_dependencies" in self._data.columns:
            all_deps = self._data.set_index("tag")["on_off_dependencies"].dropna()
            for tag, on_off_dependencies in all_deps.items():
                for dep in on_off_dependencies:
                    graph.add_dependency(tag, dep)
        self._dep_graph = graph

    @classmethod
    def from_dict(cls, data: Dict, validate: bool = True, **kwargs):
        """
        Alternative constructor. Creates new TagDict object from a dictionary. The
         dict should be structured in what pandas calls "index orientation", i.e.
         `{"tag1": {"name": "first tag", "area": "area 49", ...}, "tag2": {"name": ...}`

        Args:
            data: input dict
            validate: whether to validate the input dict. validate=False can
             lead to a dysfunctional TagDict but may be useful for testing
            **kwargs: additional keyword arguments for pandas.DataFrame.from_dict()
        """
        df = pd.DataFrame.from_dict(data, orient="index", **kwargs)
        df.index.name = "tag"

        return cls(df.reset_index(), validate)

    @classmethod
    def from_json(cls, data: str, validate: bool = True, **kwargs):
        """
        Alternative constructor. Creates new TagDict object from a json string. The
         json object should be structured in what pandas calls "index orientation", i.e.
         `{"tag1": {"name": "first tag", "area": "area 49", ...}, "tag2": {"name": ...}`

        Args:
            data: json string
            validate: whether to validate the input str. validate=False can
                      lead to a dysfunctional TagDict but may be useful for testing
            **kwargs: additional keyword arguments for pandas.read_json()
        """
        df = pd.read_json(data, orient="index", **kwargs)
        df.index.name = "tag"

        return cls(df.reset_index(), validate)

    def to_frame(self) -> pd.DataFrame:
        """
        Returns:
            underlying dataframe
        """
        data = self._data.copy()
        data["on_off_dependencies"] = data["on_off_dependencies"].apply(", ".join)
        return data

    def to_dict(self) -> Dict:
        """
        Returns the dictionary representation of the underlying dataframe in
        what pandas calls "index orientation", i.e.
        `{"tag1": {"name": "first tag", "area": "area 49", ...}, "tag2": {"name": ...}`

        Returns:
            Dict representation of underlying dataframe.
        """
        df = self.to_frame().set_index("tag")
        return df.to_dict(orient="index")

    def to_json(self, **kwargs) -> str:
        """
        Returns the json string representation of the underlying dataframe in
        what pandas calls "index orientation", i.e.
        `{"tag1": {"name": "first tag", "area": "area 49", ...}, "tag2": {"name": ...}`

        Args:
            **kwargs: additional keyword arguments for pandas.DataFrame.to_json()

        Returns:
            json string representation of underlying dataframe.
        """
        df = self.to_frame().set_index("tag")
        return df.to_json(orient="index", **kwargs)

    def _check_key(self, key: str):
        """ Check if a key is a known tag """
        if key not in self._data["tag"].values:
            raise KeyError("Tag `{}` not found in tag dictionary.".format(key))

    def _check_is_on_off(self, key: str):
        """ Check if a key is an on_off type tag """
        tag_data = self[key]
        if tag_data["tag_type"] != "on_off":
            raise TagDictError(
                "Tag `{}` is not labelled as 'on_off' tag_type in the "
                "tag dictionary.".format(key)
            )

    def __getitem__(self, key: str) -> Dict[str, Any]:
        """
        Enable subsetting by tag to get all information about a given tag.
        Args:
            key: tag name
        Returns:
            dict of tag information
        """
        self._check_key(key)
        data = self._data
        return data.loc[data["tag"] == key, :].iloc[0, :].to_dict()

    def __contains__(self, key: str) -> Dict[str, Any]:
        """
        Checks whether a given tag has a tag dict entry.
        Args:
            key: tag name
        Returns:
            True if tag in tag dict.
        """
        return key in self._data["tag"].values

    def name(self, key: str) -> str:
        """
        Returns clear name for given tag if set or tag name if not.
        Args:
            key: tag name
        Returns:
            clear name
        """
        tag_data = self[key]
        name = key
        if not pd.isnull(tag_data["name"]):
            name = tag_data["name"]
        elif not pd.isnull(tag_data["description"]):
            name = tag_data["description"].replace(" ", "_")
        name = name + "_" + str(tag_data["tag_type"])
        return name

    def dependencies(self, key: str) -> List[str]:
        """
        Get all on_off_dependencies of a given tag.
        Args:
            key: input tag
        Returns:
            list of tags that depend on input tag
        """
        self._check_key(key)
        return self._dep_graph.get_dependencies(key)

    def dependents(self, key: str) -> List[str]:
        """
        Get all dependents of a given tag.
        Args:
            key: input tag
        Returns:
            list of tags that input tag depends on
        """
        self._check_key(key)
        self._check_is_on_off(key)
        return self._dep_graph.get_dependents(key)

    def add_tag(self, tag_row: Union[dict, pd.DataFrame]):
        """
        Adds new tag row/s to the TagDict instance,
        only if and entry doesn't already exist.
        Args:
            tag_row: DataFrame or Series/dict-like object of tag row/s
        Raises:
            TagDictError if the supplied tag rows are incorrect
        """
        if not isinstance(tag_row, (dict, pd.DataFrame)):
            raise TagDictError(
                f"Must provide a valid DataFrame or "
                f"dict-like object for the tag row/s. Invalid "
                f"object of type {type(tag_row)} provided"
            )
        # Skip tags if already present in the TagDict.
        tag_data = pd.DataFrame(data=tag_row)
        tag_data.set_index("tag", inplace=True)

        tags_already_present = set(tag_data.index).intersection(set(self._data["tag"]))
        if tags_already_present:
            logger.info(
                f"[{tags_already_present}] already present in the Tag "
                f"Dictionary. Skipping."
            )
            tag_data.drop(list(tags_already_present), inplace=True)

        if not tag_data.empty:
            data = self.to_frame()
            tag_data.reset_index(inplace=True)
            data = data.append(tag_data, ignore_index=True, sort=False)

            self._data = validate_td(data) if self._validate else data

            self._update_dependency_graph()

    def select(self, filter_col: str = None, condition: Any = None) -> List[str]:
        """
        Retrieves all tags according to a given column and condition. If no filter_col
        or condition is given then all tags are returned.

        Args:
            filter_col: optional name of column to filter by
            condition: filter condition
                       if None: returns all tags where filter_col > 0
                       if value: returns all tags where filter_col == values
                       if callable: returns all tags where filter_col.apply(callable)
                       evaluates to True if filter_col is present, or
                       row.apply(callable) evaluates to True if filter_col
                       is not present
        Returns:
            list of tags
        """

        def _condition(x):

            # handle case where we are given a callable condition
            if callable(condition):
                return condition(x)

            # if condition is not callable, we will assert equality
            if condition:
                return x == condition

            # check if x is iterable (ie a row) or not (ie a column)
            try:
                iter(x)
            except TypeError:
                # x is a column, check > 0
                return x > 0 if x else False

            # x is a row and no condition is given, so we return
            # everything (empty select)
            return True

        data = self._data

        if filter_col:

            if filter_col not in data.columns:
                raise KeyError("Column `{}` not found.".format(filter_col))

            mask = data[filter_col].apply(_condition) > 0

        else:
            mask = data.apply(_condition, axis=1) > 0

        return list(data.loc[mask, "tag"])

    def get_targets(self) -> List[str]:
        """
        Retrieves all targets for different models

        Args:

        Returns:
            list of tags
        """
        data = self._data
        targets = list(data.loc[data["target"].str.contains("target", na=False), "tag"])

        return targets
