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
Functionality for creating reports based on jupyter notebook templates
"""
from pathlib import Path
from typing import Union, Dict, Any
import datetime
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
from traitlets.config import Config

from .utils import set_env_var


def _run_template(
    template_path: Union[str, Path],
    err_path: Union[str, Path] = None,
    namespace: str = "",
    kernel: str = "python3",
    timeout: int = 600,
    env: str = "local",
) -> nbformat.notebooknode.NotebookNode:
    """
    Loads and runs an ipynb template.

    Args:
        template_path: path of template notebook
        err_path: path to write to in case of execution error
        kernel: ipython kernel. Use "python3" for currently active
                virtualenv
        timeout: max run time in seconds
        env: kedro env
    Returns:
        nbconvert notebook object
    Raises:
        ValueError for wrong inputs
        CellExecutionError in case the execution fails
    """
    template_path = Path(template_path)
    if not template_path.is_file():
        raise ValueError("Template `{}` is not a file.".format(template_path))

    if err_path:
        err_path = Path(err_path)
        if err_path.exists():
            raise ValueError("Error path `{}` already exists.".format(err_path))

    with template_path.open("r") as file_:
        nb = nbformat.read(file_, as_version=4)

    target = f"{namespace}"
    nb["cells"] = [nbformat.v4.new_code_cell(f"namespace={repr(target)}")] + nb["cells"]

    epp = ExecutePreprocessor(kernel_name=kernel, timeout=timeout)

    with set_env_var("KEDRO_ENV", env):
        try:
            epp.preprocess(nb, {"metadata": {"path": str(template_path.parent)}})
        except CellExecutionError as cell_ex:
            if err_path:
                with err_path.open("w") as file_:
                    nbformat.write(nb, file_)
                raise RuntimeError(
                    "Notebook execution failed. See {} for more details.".format(
                        err_path
                    )
                ) from cell_ex

            raise cell_ex

    return nb


def create_ipynb_report(
    template_path: Union[str, Path],
    out_path: Union[str, Path],
    namespace: str = "",
    kernel: str = "python3",
    timeout: int = 600,
    env: str = "local",
):
    """
    Creates an ipynb report from an ipynb template.

    Args:
        template_path: path of template notebook
        err_path: path (file name) to write to
        kernel: ipython kernel. Use "python3" for currently active
                virtualenv
        timeout: max run time in seconds
        env: kedro env
    Raises:
        ValueError for wrong inputs
    """
    out_path = Path(out_path)
    if out_path.exists():
        raise ValueError("Output path `{}` already exists.".format(out_path))
    if not out_path.parent.is_dir():
        raise ValueError(
            "Parent folder of output path `{}` does not exists.".format(out_path)
        )

    err_path = Path(str(out_path) + "_failed.ipynb")
    processed = _run_template(
        template_path, err_path, namespace, kernel=kernel, timeout=timeout, env=env
    )

    with out_path.open("w") as file_:
        nbformat.write(processed, file_)


def create_html_report(params: Dict, kedro_env: str, *wait_on: Any):
    """
    Creates an html report from an ipynb template.

    Args:
        params: html report parameters
        kedro_env: kedro env
        wait_on: any catalog entries to wait on - not used in method
    Raises:
        ValueError for wrong inputs
    """

    template_path = params["template_path"]
    output_dir = params["output_dir"]
    report_name = params["report_name"]
    namespace = params["namespace"]

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    out_path = (
        Path(output_dir).joinpath(f"{report_name}_{timestamp_str}.html")
        if params.get("timestamp", False)
        else Path(output_dir).joinpath(f"{report_name}.html")
    )
    out_path = Path(out_path)
    if out_path.exists():
        raise ValueError("Output path `{}` already exists.".format(out_path))
    if not out_path.parent.is_dir():
        raise ValueError(
            "Parent folder of output path `{}` does not exists.".format(out_path)
        )

    err_path = Path(str(out_path) + "_failed.ipynb")
    processed = _run_template(
        template_path,
        err_path,
        namespace,
        kernel=params.get("kernel", "python3"),
        timeout=params.get("timeout", 600),
        env=kedro_env,
    )

    c = Config()
    # we remove input and output prompts by default
    c.HTMLExporter.exclude_input_prompt = True
    c.HTMLExporter.exclude_output_prompt = True

    if params.get("remove_code", False):
        c.HTMLExporter.exclude_input = True

    html_exporter = HTMLExporter(c)
    body, _ = html_exporter.from_notebook_node(processed)

    with out_path.open("w") as file_:
        file_.write(body)
