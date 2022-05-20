# Data Validation
This module helps identify anomalies and outliers in the dataset. It is based on kedro GE framework and has custom expectations that can be used.
## How to get started?
The following steps will help get started with setting up data validation.
### Installing Kedro GE
#### Step 1: Install kedro GE
You can install kedro-great-expectations using pip. 
```bash
pip install optimus/packages/kedro_great_expectations-0.3.0-py3-none-any.whl
```
> .whl file for kedro_great_expectations is also available on [box](https://mckinsey.box.com/v/kedro-great-expectations)

##### Default Rules
Please refer to this [link](https://docs.greatexpectations.io/en/v0.4.4/glossary.html) for all the available rules.
#### Step 2: Using kedro GE in pipeline
Modify your src/optimus_pkg/ins_data_assets/run.py to the following to enable validation on kedro run
```python
from kedro.context import KedroContext
from kedro_great_expectations import GreatExpectationsMixin
 
class ProjectContext(GreatExpectationsMixin, KedroContext):
    # refer to sample config in optimus/pipeline/conf/base/kedro_ge.yml
    ge_config_key = "kedro_ge.yml"   # optional, defaults to this value
    ### ...
```
#### Step 3: Setting up and configuring great expectations
When kedro GE is initiated, it generates a `kedro_ge.yml` configuration file  in the `conf/base` folder. This can be done by running the following command
```bash
kedro ge init
```
This file can be configured to suit the project needs. The class path for custom expectations developed for OptimusAI is included in this file.
For more information on how to configure, please refer to [GE documentation](https://one.quantumblack.com/docs/alchemy/kedro_great_expectations/03_user_guide/01_configuration.html).

#### Step 4: Create a GE suite.
The following commands will help create an empty GE suite for each dataset. Make sure you are in the pipeline folder before executing the commands.
 
``` bash
cd pipeline
kedro ge generate <dataset_name> --empty
kedro ge edit <dataset_name>
``` 

This will open a jupyter notebook `dataset_name.ipynb` for editing.
 
#### **Step 5: Build your Expectation Suite**
OptimusAI has built some custom expectations that can be used in addition to those provided by GE. These can be found in the `great_expectations_utils.py` file. The custom expectation class and its methods are detailed in the `adtk_custom_expectation.py` file.
Simply copy paste the desired method into the notebook. The below example implements Anomaly detection using quantiles.

``` python
from optimus_pkg.data_validation.great_expectations_utils import *
params = context.params

# Custom Expectation - Quantile anomaly detection
validate_column_quantile_anomaly(batch, params)
```
The parameter file for data validation module is located at `<my_project>/pipeline/conf/base/pipelines/validate/parameters.yml`

## Custom Expectations
Currently, OptimusAI supports two types of anomaly detection. 
- Rule based anomaly detection
- Model based advanced anomaly detection

All of them have been implemented using the Anomaly Detection ToolKit ([ADTK](https://adtk.readthedocs.io/en/stable/index.html)) package.

### Rule based anomaly detection
The following methods detect anomalies using set rules to detect anomalies.
1. Level Shift Anomaly Detection: `create_level_shift_expectation`
This detects level shifts in the dataset by comparing values from two time windows.
2. Quantile Anomaly Detection: `validate_column_quantile_anomaly`
This detects anomalies based on quantiles of historical data
3. Persist Anomaly Detection: `validate_column_persist_anomaly`
This detects anomalies based on values in a preceding time period.

### Advanced Anomaly detection 
Sometimes, it is difficult to detect anomalies based on simple rules. Model based anomaly detection can help solve this issue. The following methods are currently available. 
1. Isolation Forest: `validate_multi_dimension_isolationforest_anomaly`
This method identifies time points as anomalous based isolation forest technique. This is a tree based technique and is highly effective in high dimensional data. 
2. KMeans Clustering: `validate_multi_dimension_cluster_anomaly`
This method identifies anomalies based on clustering historical data

## FAQ
### 1. Can I add my own expectation?
Yes, you can create your own expectation. 
1. Go to ./pipeline/src/optimus_pkg/data_validation/adtk_custom_expectation.py
2. Add your function to the class `CustomADTKExpectations`.
3. Include your function in the GE utils file, i.e., `great_expectations_utils.py`. 
4. Call this function when creating your GE suite through the jupyter notebook generated for your dataset.

### 2. What are decorators? How is it used here?
Decorators are callable objects that add new functionality to existing objects without modifying its structure. GE provides high-level decorators that help convert our custom functions into a fully-fledged expectation.
We use `column_aggregate_expectation` decorator from class `MetaPandasDataset`, other options include . For more information on them, refer to the [GE docs](https://docs.greatexpectations.io/en/latest/autoapi/great_expectations/dataset/index.html#great_expectations.dataset.MetaPandasDataset). 

### 3. How should I configure my expectation file?
You can configure how validation works for your datasets by using the following [config schema](https://one.quantumblack.com/docs/alchemy/kedro_great_expectations/03_user_guide/01_configuration.html)

### 4. Something is not right. How can I get in touch?
Please raise an issue [here](https://git.mckinsey-solutions.com/opm/optimus/issues/new/choose). Alternatively, get in touch via slack [#optimus](https://mckinsey-client-cap.slack.com/archives/C9S1RM6SX).