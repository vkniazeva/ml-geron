# Notes to the chapter 1

## Building a new ML project

Usually all ML projects can be built based on the following plan:

1. Look at the big picture 
   * Business scope: task, target, application, focus 
   * Current situation: process, costs, errors 
   * ML model: task type, algorithm, data update, quality criteria
2. Get the data:
    * checking dependencies, distribution
    * identifying the most important features
    * validate: data is representative?
    * cleaning up data: missing values, normalizing etc.
    * applying stratification
3. Discover and visualize the data to gain insights.
4. Prepare the data for Machine Learning algorithms.
5. Select a model and train it.
6. Fine-tune your model.
7. Present your solution.
8. Launch, monitor, and maintain your system.


## Glossary

- Data snooping bias—recognizing data patterns, existing only in a concreate dataset
and disappearing with the first check on new data.
- Stratified sampling is a method that ensures the sample contains representative 
proportions of different classes/groups compared to the overall population. 
It is typically done by dividing the population into strata (subgroups) 
and then sampling from each stratum.


## Useful Python insights

- DataFrame—the primary data structure in pandas, 
representing a two-dimensional table similar to 
an Excel worksheet. It consists of rows and columns, 
where each column can hold a different data type (e.g., 
strings, integers, dates).
- head()—method used to quickly inspect the beginning of a 
DataFrame by returning the first 5 rows.
- info()—method used to get a short summary about data (number of rows, 
each attribute type, None values number)
- value_counts() - method for counting non-numerical values appearance 
- describe()—method for getting a statistical summary for numerical attributes (
count, mean, std. min, percentiles, max)
- hist() - matplotlit.pyplot - drawing a histogram: bin - detalization (higher -> more details),
figsize - graph size (i.e. 20 - width, 15 - high)

![histogram.png](../assets/chapter_1/histogram.png)

- train_test_split() - sklearn.model_selection - supports a dataset splitting to
a train and test ones. Params: dataset, test_size, random_state (repetitive split)

- cut() - pd -  a pandas function for binning continuous numerical data into intervals. 
Parameters: data array, bins (number of intervals or bin edges),
labels (optional names for the intervals).

![histogram_income_cat.png](../assets/chapter_1/histogram_income_cat.png)

- loc - pd - a pandas function for selecting data by labels or by boolean arrays

- StratifiedShuffleSplit - sklearn.model_selection - creates train/test splits while preserving 
the class distribution of the target variable. The split() method returns indices for stratified 
random shuffling and partitioning of the data. Params StratifiedShuffleSplit: n_splits - number of sets to generate, 
test_size, random_state. Split() expects dataset and column (array)

- drop() - params: axis=1 for columns and 0 for raws, inplace = True (the same set)

- select_dtypes() - selects only datatypes from given arguments i.e. include=["int64", "float64"] or include=[np.number],
where np.number stands for all numerical types 

- scatter_matrix - pandas.plotting - scatter matrix for selected attributes, where as diagonal histograms and 
scatter plots for pair relationships in other cells are plotted. Params: a list of selected numeric features, figsize - display size 
