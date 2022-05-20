import pandas as pd
from kedro.io import DataCatalog
from kedro.extras.datasets.pickle import PickleDataSet
from kedro.extras.datasets.pandas import CSVDataSet
import datetime

def get_active_models(parameters: dict) -> [str]:
    """
    Get all active models, in order to skip reading information
    that is not fresh.
    Args:
        parameters: Dictionary containing all parameters in Optimus.
    Returns:
        models: List of strings with all model names.
    """
    models = [x.split('.')[0] for x in parameters.keys() if x.endswith('model_input')]

    models = [x for x in models if x not in parameters['skip_models']]
    return models

def generate_in_memory_catalog(parameters: dict) -> DataCatalog:
    """
    Generate in-memory Kedro Catalog. Maybe this is not the best way to proceed,
    but at this point, is the best way to proceed.
    Args:
        parameters: Dictionary containing all parameters of the project.
    Returns:
        catalog: DataCatalog object with specifics files in memory to be load.
    """
    models = get_active_models(parameters)
    
    train_models = {f"{model}.train_model": PickleDataSet(
        filepath= f"data/06_models/{model}/train_model.pkl") for model in models
        }
    train_set_metrics =  {f"{model}.train_set_metrics": CSVDataSet(
        filepath = f"data/08_reporting/{model}/train_set_metrics.csv",
        load_args = dict(index_col= 0),
        save_args = dict(index = True)
        ) for model in models
        }
    test_set_metrics =  {f"{model}.test_set_metrics": CSVDataSet(
        filepath = f"data/08_reporting/{model}/test_set_metrics.csv",
        load_args = dict(index_col= 0),
        save_args = dict(index = True)
        ) for model in models
        }

    master_dict = {}
    master_dict.update(train_models)
    master_dict.update(train_set_metrics)
    master_dict.update(test_set_metrics)

    catalog = DataCatalog(
        master_dict
    )
    
    return catalog

def get_best_params(model: str, catalog: DataCatalog) -> pd.DataFrame:
    """
    Get the best parameters from the GridSearch made it 
    for the latest XGBoost regressor, trained for an specific model.
    Args:
        model: string with the model name.
    Returns:
        best_params: DataFrame containing the best parameters by column.
    """
    train_model = f'{model}.train_model'
    best_params = catalog.load(train_model).steps[1][1].get_xgb_params()
    best_params = {k:[v] for k,v in best_params.items()}
    return pd.DataFrame.from_dict(best_params)

def get_params_gs(model: str, parameters:dict) -> pd.DataFrame:
    """
    Get the original grid of parameters of the GridSearch 
    for the latest XGBoost regressor.
    Args:
        model: string with the model name.
    Returns:
        best_params: DataFrame containing the list of parameters by column.
    """
    train_model = f'{model}.train_model'
    max_params_gridsearch = parameters[train_model]['tuner']['kwargs']['param_distributions']
    max_params_gridsearch = {k.split('__')[1]:[v] for k,v in max_params_gridsearch.items()}
    return pd.DataFrame.from_dict(max_params_gridsearch)

def compare_metrics_df(best_params_df: pd.DataFrame
                       , gs_params_df: pd.DataFrame
                       , cols_to_look: [str]) -> pd.DataFrame:
    """
    Merge and create new columns for checking if the best parameters
    from the GridSearch was the highest values of the grid, in order
    to generate some idea of possible overfitting.
    Args:
        best_params_df: DataFrame containing the best parameters of the trained
        model.
        gs_params_df: DataFrame containing the grid of parameters, each list per
        columns.
        cols_to_look: List containing the parameters name to check.
    Returns:
        params_df: Merged DataFrame with extra boolean columns pointing
        if the best parameter is the greates in its grid.
    """
    compare_params = best_params_df.merge(gs_params_df, left_index = True, right_index = True, )
    for col in cols_to_look:
        compare_params[f'max_{col}'] = compare_params.apply(lambda row: row[col]==max(row[f'gs_{col}']), axis = 1)
    params_df = compare_params[[x for x in compare_params.columns if not x.startswith('gs_')]]
    return params_df

def get_params_df(model: str, params_to_look: [str], catalog: DataCatalog, parameters: dict) -> pd.DataFrame:
    """
    Generate a single DataFrame containing metrics about parameters
    chose by GridSearch, and append a first column for the Model name.
    Args:
        model: String of the model to review.
        params_to_look: List containing the parameters of of the GridSearch
        to inspect.
    Returns:
        params_df: DataFrame with parameters metrics.
    """
    model_df = pd.DataFrame([model], columns = ['model'])
    best_params_df = get_best_params(model, catalog)[params_to_look]
    gs_params_df = get_params_gs(model, parameters)[params_to_look]
    gs_params_df.columns = [f'gs_{x}' for x in gs_params_df.columns]
    
    params_df = compare_metrics_df(best_params_df, gs_params_df, params_to_look)
    
    params_df = model_df.merge(params_df, left_index = True, right_index = True, )
    return params_df

def train_test_metrics(model: str, metrics_to_look: [str], catalog:DataCatalog) -> pd.DataFrame:
    """
    Grab the saved metrics for the training and test set, and generate
    a single DataFrame containing those ones, and some extra columns that 
    indicate some changes between the requested metrics.
    Args:
        model: Model name.
        metrics_to_look: List containing the metrics to compare 
        e.g MAPE, R2, RMSE, etc.
    Returns:
        compare_metrics: DataFrame with all requested information.
    """
    train_ = catalog.load(f'{model}.train_set_metrics').T[metrics_to_look]
    train_.columns = [f'train_{x}' for x in train_.columns]
    test_ = catalog.load(f'{model}.test_set_metrics').T[metrics_to_look]
    test_.columns = [f'test_{x}' for x in test_.columns]
    
    ##TODO: skip the hardcoded stuff
    compare_metrics = train_.merge(test_, left_index = True, right_index = True, )
    if 'mape' in metrics_to_look:
        compare_metrics['mape_change'] = (compare_metrics.test_mape - compare_metrics.train_mape)/compare_metrics.train_mape*100
    if 'r2' in metrics_to_look:
        compare_metrics['r2_change'] = (compare_metrics.test_r2 - compare_metrics.train_r2)
    compare_metrics.reset_index(inplace=True, drop = True)
    return compare_metrics

def get_all_metrics(parameters: dict) -> pd.DataFrame:
    """
    Concatenate all functions created before in a single node
    due we need to do gather all the metrics in a single step.
    Args:
        parameters: Dictionary containing all parameters of the project.
    Returns:
        all_models: DataFrame containing all needed metrics.
    """
    catalog = generate_in_memory_catalog(parameters)
    metrics_to_look = parameters["model_reporting"]["metrics_to_look"]
    params_to_look = parameters["model_reporting"]["params_to_look"]
    models = models = get_active_models(parameters)
    all_models = pd.DataFrame()
    for model in models:
        model_df = pd.DataFrame([model], columns = ['model'])
        params_df = get_params_df(model, params_to_look, catalog, parameters)
        compare_metrics = train_test_metrics(model, metrics_to_look, catalog)
        int_ = params_df.merge(compare_metrics, left_index = True, right_index = True, )
        all_models = all_models.append(int_)
    all_models['date'] = datetime.datetime.now().date()
    return all_models