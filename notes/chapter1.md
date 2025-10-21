# Notes to the chapter 1

## Building a new ML project

Usually all ML projects can be built based on the following plan:

1. Look at the big picture 
   2. Business scope: task, target, application, focus
   3. Current situation: process, costs, errors
   4. ML model: task type, algorithm, data update, quality criteria
2. Get the data.
3. Discover and visualize the data to gain insights.
4. Prepare the data for Machine Learning algorithms.
5. Select a model and train it.
6. Fine-tune your model.
7. Present your solution.
8. Launch, monitor, and maintain your system.


## Glossary

- Data snooping bias—recognizing data patterns, existing only in a concreate dataset
and disappearing with the first check on new data.
- 

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

![histogram.png](assets/histogram.png)