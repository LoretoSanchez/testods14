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
io classes for TagDict
"""
import copy
from typing import Any, Dict

from kedro.extras.datasets.pandas import ExcelDataSet, CSVDataSet
from kedro.extras.datasets.text import TextDataSet

# CSVLocalDataSet, TextLocalDataSet
from kedro.io.core import Version

from .tag_dict import TagDict


class TagDictCSVLocalDataSet(CSVDataSet):
    """ Loads and saves a TagDict object from/to csv """

    def _load(self) -> TagDict:
        df = super()._load()
        return TagDict(df)

    def _save(self, data: TagDict) -> None:
        df = data.to_frame()
        super()._save(df)


class TagDictJSONLocalDataSet(TextDataSet):
    """ Loads and saves a TagDict object from/to json """

    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        json_load_args: Dict[str, Any] = None,
        json_save_args: Dict[str, Any] = None,
        version: Version = None,
    ) -> None:
        """
        Creates a new instance of ``TagDictJSONLocalDataSet``.
        See `TextLocalDataSet` for most parameters.
        """
        super().__init__(filepath, load_args, save_args, version)
        self._json_load_args = copy.deepcopy(json_load_args) or {}
        self._json_save_args = copy.deepcopy(json_save_args) or {}

    def _load(self) -> TagDict:
        json_str = super()._load()
        return TagDict.from_json(json_str, **self._json_load_args)

    def _save(self, data: TagDict) -> None:
        json_str = data.to_json(**self._json_save_args)
        super()._save(json_str)


class TagDictExcelLocalDataSet(ExcelDataSet):
    """
    Loads and saves a TagDict object from/to excel

    To load from a specific sheet, add "sheet_name" to the
    "load_args" in your catalog entry. To save to a specific
    sheet, add "sheet_name" to the "save_args" in your catalog entry.

    """

    def _load(self) -> TagDict:
        df = super()._load()
        return TagDict(df)

    def _save(self, data: TagDict) -> None:
        df = data.to_frame()
        super()._save(df)
