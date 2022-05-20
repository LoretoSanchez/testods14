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
Tag Dict Validation
"""
import pandas as pd

REQUIRED_COLUMNS = {
    "tag",
    "name",
    "tag_type",
    "data_type",
    "unit",
    "range_min",
    "range_max",
    "op_min",
    "op_max",
    "on_off_dependencies",
    "derived",
}

UNIQUE = {
    "tag",
}

COMPLETE = {
    "tag",
}

KNOWN_VALUES = {
    "tag_type": {"input", "output", "state", "control", "on_off", "reaction"},
    "data_type": {"numeric", "categorical", "boolean", "datetime"},
}
NON_ZERO = {"range_max", "max_delta"}

# tags are checked for whether they break any of the below rules
# captured as rule - explanation
ILLEGAL_TAG_PATTERNS = [
    (r"^.*,+.*$", "no commas in tag"),
    (r"^\s.*$", "tag must not start with whitespace character"),
    (r"^.*\s$", "tag must not end with whitespace character"),
]


class TagDictError(Exception):
    """ Tag Dictionary related exceptions """


def validate_td(  # pylint:disable=too-many-locals,too-many-branches
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validates a tag dict dataframe.

    Args:
        data: tag dict data frame

    Returns:
        validated dataframe with comma separated values parsed to lists
    """
    data = data.copy()

    # check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_cols:
        raise TagDictError(
            "The following columns are missing from the input dataframe: {}".format(
                missing_cols
            )
        )

    # check completeness
    for col in COMPLETE:
        if data[col].isnull().any():
            raise TagDictError("Found missing values in column `{}`".format(col))

    # check duplicates
    for col in UNIQUE:
        duplicates = data.loc[data[col].duplicated(), col]
        if not duplicates.empty:
            raise TagDictError(
                "The following values are duplicated in column `{}`: {}".format(
                    col, list(duplicates)
                )
            )

    # check that tag names don't contain invalid characters
    for (pattern, rule) in ILLEGAL_TAG_PATTERNS:
        matches = data.loc[data["tag"].str.match(pattern), "tag"]
        if not matches.empty:
            raise TagDictError(
                "The following tags don't adhere to rule `{}`: {}".format(
                    rule, list(matches)
                )
            )

    # valid max values
    for col in NON_ZERO:
        invalid = set(data[col].dropna()).intersection({0})
        if invalid:
            raise TagDictError(
                "Found invalid entries in column {}:. Must be one different from zero".format(
                    col
                )
            )

    invalid = data[
        [
            "range_min",
            "range_max",
        ]
    ].dropna()
    invalid["diff"] = invalid["range_max"] - invalid["range_min"]
    if min(invalid["diff"]) <= 0:
        raise TagDictError(
            "Found invalid entries in column range_min or range_max:. max must be higher than min"
        )

    invalid = data[
        [
            "op_min",
            "op_max",
        ]
    ].dropna()
    invalid["diff"] = invalid["op_max"] - invalid["op_min"]
    if min(invalid["diff"]) <= 0:
        raise TagDictError(
            "Found invalid entries in column op_min or op_max:. max must be higher than min"
        )

    # valid restricted values
    for col, known_vals in KNOWN_VALUES.items():
        invalid = set(data[col].dropna()) - known_vals
        if invalid:
            raise TagDictError(
                "Found invalid entries in column {}: {}. Must be one of: {}".format(
                    col, invalid, known_vals
                )
            )

    # check on_off_dependencies
    all_tags = set(data["tag"])
    on_off_tags = set(data.loc[data["tag_type"] == "on_off", "tag"])

    on_off_dependencies = data["on_off_dependencies"]
    if not isinstance(on_off_dependencies.iloc[0], list):
        on_off_dependencies = (
            data["on_off_dependencies"]
            .fillna("")
            .apply(lambda x: [xx.strip() for xx in str(x).split(",") if xx.strip()])
        )
    for idx, deps in on_off_dependencies.items():
        not_in_tags = set(deps) - all_tags
        not_in_on_off = set(deps) - on_off_tags
        if not_in_tags:
            raise TagDictError(
                "The following on_off_dependencies of {} are not known tags: {}".format(
                    data.loc[idx, "tag"], not_in_tags
                )
            )

        if not_in_on_off:
            raise TagDictError(
                "The following on_off_dependencies of {} are not labelled as "
                "on_off type tags: {}".format(data.loc[idx, "tag"], not_in_on_off)
            )

    data["on_off_dependencies"] = on_off_dependencies

    return data
