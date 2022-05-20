## Transformers

### What are Transformers?
Transformers are sklearn classes that do some sort of transformation to the input data. They are represented by classes that have a `fit` method (learning the model parameters), and a `transform` method (performing some transformation of the input data.

In Optimus, we chain Transformers to perform dynamic calculation of features not conducted in the static steps. Transformers that may be used include:

* Imputers
* PCA
* Feature Selection
* One Hot Encoding
* ...

Transformers are used in combination with sklearn [Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).

More details about Transformers that sklearn includes can be found [here](https://scikit-learn.org/stable/data_transforms.html).

### What are Transformer Pipelines?

Pipelines are objects that consist of a series of steps (Transformers objects), with the final step being an estimator.

Pipelines are useful for establishing consistency between the training model and the inference model, capturing all the intermediate steps appropriately.

Furthermore, we can have multiple estimators as steps in Pipelines, and when using GridSearch we gain flexibility to for tuning from the hyperparameters in the one estimator to that of the whole pipeline.


### Custom Transformers included in Optimus
* `optimus_pkg.transformers.SelectColumns`: Transfomer to select columns from input data using a list or regex matching string

```python
SelectColumns(regex="imputed_*")
```
* `optimus_pkg.transformers.DropColumns`: Transfomer to drop columns from input data using a list or regex matching string

```python
DropColumns(items=["col1", "col2"])
```
* `optimus_pkg.transformers.DropAllNull`: Transfomer to drop columns from input data where all values in the column are null

```python
DropAllNull()
```
* `optimus_pkg.transformers.NumExprEval`: Transfomer to create columns at runtime using valid [NumExpr](https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/user_guide.html#supported-operators) string expressions. Expressions are carried out sequentially.

```python
NumExprEval(exprs=["C=A+B", "E=sin(A)+tan(C)/cos(B)"])
```
* `optimus_pkg.transformers.SklearnTransform`: That wraps a default Sklearn Transformer to be compatible with the Optimus training pipeline. Default sklearn Transformers tend to return transformed results as `np.ndarray`. This custom Transformer ensures that DataFrames are returned. It will retain the original column names.

```python
SklearnTransform(transformer=SimpleImputer())
```
* `optimus_pkg.transformers.MissForestImpute`: Transfomer that uses a Random Forest Based Imputation method. [source](https://github.com/epsilon-machine/missingpy)

```python
MissForestImpute(max_iter=5, n_estimators=30)
```

### How to Write Custom Transformers
1. Place your Custom Transformer in `optimus_pkg/transformers`, and inherit from `optimus_pkg.transformers.Transformer`.
2. Implement a `.fit` function for this Transformer that learns the state of the data to be processed. Ensure you call the `.check_x()` function checking if the passed `x` is a Pandas DataFrame.
3. Implement the `.transform` function. This function does the transformation/data manipulation of the original DataFrame. Similarly call the `.check_x()` function at the beginning of the routine. You should be returning a Pandas DataFrame.

Examples of Custom Transformers can be seen above.