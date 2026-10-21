# Course Notes: Exploratory Data Analysis for Machine Learning

These notes summarize key concepts from IBM's *Exploratory Data Analysis for Machine Learning* course on Coursera.

## 1. Foundations of Artificial Intelligence and Machine Learning

### Artificial Intelligence, Machine Learning, and Deep Learning

- **Artificial Intelligence (AI):** Systems designed to perform tasks associated with intelligent behavior, including reasoning, perception, and adaptation.
- **Machine Learning (ML):** A subset of AI that learns patterns from data rather than relying entirely on explicitly programmed rules.
- **Deep Learning (DL):** A subset of ML that uses multi-layer neural networks to learn representations from data.

### Types of Machine Learning

- **Supervised learning:** Uses labeled data to predict a target or label. Examples include spam detection and flower classification.
- **Unsupervised learning:** Uses unlabeled data to discover patterns or structures, such as customer segmentation.

### Historical Development and Applications

The course discusses AI milestones such as the Turing Test, the Dartmouth Conference, the perceptron, expert systems, and advances in deep learning. It also describes periods of reduced AI investment known as *AI winters*.

Modern AI progress has been supported by larger datasets, improved computing hardware, cloud infrastructure, and advances in neural networks.

Applications include healthcare, finance, transportation, manufacturing, agriculture, retail, education, recommendation systems, computer vision, and natural language processing (NLP).

## 2. Machine Learning Workflow

A typical machine learning workflow includes:

1. Define the problem statement.
2. Collect or retrieve relevant data.
3. Explore the dataset and assess its quality.
4. Clean and preprocess the data.
5. Engineer and transform features.
6. Build and validate a model.
7. Use the results to support decisions and deployment.

### Basic Terminology

- **Feature:** An input variable used to describe an observation or make a prediction.
- **Target variable:** The outcome the model is intended to predict.
- **Observation:** A single record or row in a dataset.
- **Label:** The target value associated with an observation in supervised learning.

## 3. Data Retrieval

Data can be retrieved from files, databases, and external services.

### CSV and JSON

- **CSV:** A common tabular format that can be loaded using Pandas `read_csv()`.
- **JSON:** A structured data format frequently used by APIs and document-oriented databases. Pandas provides `read_json()` and `to_json()`.

### SQL and NoSQL Databases

Relational databases organize data into tables and can be queried using SQL. Examples discussed in the course include SQLite, MySQL, PostgreSQL, and Microsoft SQL Server.

Python tools such as `sqlite3`, SQLAlchemy, and database-specific connectors can be used to connect to databases. Pandas `read_sql()` can load query results into a DataFrame.

NoSQL databases use non-relational data models. MongoDB and its Python connector, `pymongo`, are examples mentioned in the course.

### Data Inspection

Before analysis, inspect the retrieved data to understand its structure, columns, data types, and available records. SQL queries can also filter, group, aggregate, and sort records before loading them into Python.

## 4. Data Cleaning and Data Quality

Data quality affects the reliability of analysis and machine learning. Incorrect, inconsistent, duplicated, or incomplete records can lead to misleading results.

### Common Data Quality Issues

- Duplicate records.
- Inconsistent category names or spelling.
- Missing values.
- Incorrect data types.
- Outliers and unusual observations.
- Irrelevant identifiers or columns.

### Handling Missing Values

Possible strategies include:

- Removing rows or columns when justified.
- Imputing values using an appropriate statistic or method.
- Treating missingness as a separate category when meaningful.

The appropriate approach depends on the amount of missing data, the variable, and the reason values are missing. Removing observations may reduce sample size or introduce bias.

### Outlier Detection

An outlier is an observation that differs substantially from much of the dataset. It may represent an error, an unusual but valid case, or an informative event.

Common approaches include:

- Histograms and box plots.
- The interquartile range (IQR) rule.
- Examining residuals from a fitted model.

The IQR rule flags values below \(Q1 - 1.5 \times IQR\) or above \(Q3 + 1.5 \times IQR\) for investigation. These are potential outliers, not automatically invalid records.

Outliers should be handled according to context rather than removed solely because they are statistically unusual.

## 5. Exploratory Data Analysis (EDA)

EDA uses statistical summaries and visual methods to understand a dataset before further analysis or modeling.

### Objectives of EDA

- Understand the structure and quality of the data.
- Summarize distributions and central tendencies.
- Identify missing values and unusual observations.
- Explore relationships between variables.
- Discover patterns that may guide further analysis.

### Descriptive Statistics

Useful summaries include:

- Mean and median.
- Minimum and maximum.
- Quantiles and range.
- Standard deviation.
- Category frequencies.
- Correlations between numeric variables.

Pandas methods such as `describe()`, `value_counts()`, `groupby()`, and `agg()` support these tasks.

### Sampling

Sampling can make analysis more manageable when working with large datasets.

- **Random sampling:** Selects observations randomly.
- **Sampling with replacement:** Allows the same observation to be selected more than once.
- **Stratified sampling:** Maintains the proportions of specified categories, which can be useful for datasets with imbalanced outcomes.

Pandas provides the `sample()` method for drawing subsets of data.

## 6. Data Visualization

Visualization helps reveal patterns, distributions, differences, and potential anomalies that may not be obvious from summary statistics alone.

### Python Visualization Libraries

- **Matplotlib:** Flexible, customizable plotting.
- **Pandas plotting:** Convenient plotting methods for DataFrames.
- **Seaborn:** Statistical visualizations built on Matplotlib with a higher-level interface.

### Common Plot Types

| Plot | Main Purpose |
|---|---|
| Histogram | Examine the distribution of a numeric variable |
| Scatter plot | Explore relationships between two numeric variables |
| Box plot | Compare distributions and identify potential outliers |
| Pair plot | Explore pairwise relationships among several features |
| Joint plot | Examine a relationship alongside marginal distributions |
| FacetGrid | Compare patterns across categories in separate panels |

Plots should have clear titles, axis labels, and legends where appropriate.

### Practical Example: Iris Dataset

The course uses the Iris dataset to demonstrate data loading, cleaning category labels, calculating summary statistics, grouping observations by species, and visualizing sepal and petal measurements.

These examples show how visual analysis can reveal differences between groups and relationships among numeric features.

## 7. Feature Engineering and Variable Transformation

Feature engineering involves creating, selecting, or transforming input variables to make them more useful for a machine learning model.

### Feature Selection

Feature selection identifies which variables should be included in a model. Unique identifiers may be removed when they do not provide useful predictive information.

The relevance of a feature depends on the problem and the modeling approach.

### Encoding Categorical Variables

Machine learning algorithms often require categorical variables to be represented numerically.

- **Binary encoding:** Represents two-category variables using values such as 0 and 1.
- **One-hot encoding:** Creates binary columns for categories without implying a natural order.
- **Ordinal encoding:** Assigns numeric values to categories with a meaningful order.

The choice depends on whether the categories are nominal or ordinal.

### Feature Scaling

Scaling places numeric features on comparable scales.

- **Standard scaling:** Subtracts the mean and divides by the standard deviation.
- **Min-max scaling:** Maps values to a specified range, commonly 0 to 1.
- **Robust scaling:** Uses the median and interquartile range, making it less sensitive to outliers.

Scaling is particularly relevant for scale-sensitive algorithms, such as k-nearest neighbors.

### Transformations for Skewed Data

Logarithmic transformations, `log1p()`, and Box-Cox transformations can help address skewed distributions.

Polynomial features, such as \(x^2\) and \(x^3\), allow a model that is linear in its parameters to represent nonlinear relationships. Interaction features represent combined effects between variables.

The course also demonstrates group-based deviation features, calculated by comparing a value with its category-specific mean and standard deviation.

### Practical Example: Ames Housing Dataset

The Ames Housing examples cover inspecting a dataset, investigating unusual living-area values, removing non-predictive identifiers, examining skewness, applying log transformations, handling missing values, and generating polynomial and interaction features.

## 8. Statistical Inference and Probability Distributions

Statistical inference uses information from a sample to draw conclusions about a population while accounting for uncertainty.

### Estimation

An estimate uses sample data to approximate a population parameter. Standard errors and confidence intervals help describe the uncertainty associated with estimates.

### Parametric and Non-parametric Approaches

- **Parametric methods:** Use a specified model form described by a finite set of parameters.
- **Non-parametric methods:** Make fewer assumptions about a particular distribution.
- **Maximum Likelihood Estimation (MLE):** Estimates model parameters by maximizing the likelihood of the observed data under the model.

### Probability Distributions

- **Uniform distribution:** Outcomes in a specified range have equal probability.
- **Normal distribution:** A symmetric, bell-shaped distribution centered around its mean.
- **Log-normal distribution:** A positive-valued variable whose logarithm follows a normal distribution.
- **Exponential distribution:** Often used to model waiting times between events.
- **Poisson distribution:** Models event counts over a specified interval using a rate parameter.

### Frequentist and Bayesian Statistics

Frequentist statistics interprets probability through repeated observations and treats population parameters as fixed.

Bayesian statistics represents uncertainty about parameters or hypotheses using probability distributions. It combines prior information with observed data to obtain a posterior distribution.

## 9. Hypothesis Testing

Hypothesis testing provides a formal framework for evaluating claims about a population using sample evidence.

### Core Concepts

- **Null hypothesis (\(H_0\)):** The baseline claim being evaluated.
- **Alternative hypothesis (\(H_1\) or \(H_a\)):** The competing claim.
- **Test statistic:** A quantity calculated from sample data to evaluate evidence against the null hypothesis.
- **Significance level (\(\alpha\)):** A threshold selected before testing to guide the decision.

### P-values

A p-value is the probability, assuming the null hypothesis is true, of observing a test result at least as extreme as the one obtained.

A result is commonly called statistically significant when its p-value is below the selected significance level. A p-value is not the probability that the null hypothesis is true.

### Type I and Type II Errors

- **Type I error:** Rejecting a true null hypothesis.
- **Type II error:** Failing to reject a false null hypothesis.
- **Statistical power:** The probability of correctly rejecting a false null hypothesis.

The significance threshold affects the balance between Type I and Type II errors.

### Likelihood Ratios and Multiple Testing

A likelihood ratio compares how likely observed data are under two competing hypotheses or models.

When many hypotheses are tested, the probability of obtaining at least one false positive increases. Multiple-testing procedures, including p-value adjustment methods, help address this issue.

The course also discusses regression F-statistics and Bayesian hypothesis testing.

## 10. Correlation and Causation

**Correlation** describes an association between variables. It can be useful for identifying patterns and making predictions.

**Causation** means that a change in one variable directly affects another.

An observed correlation may arise because:

- One variable affects the other.
- The direction of influence is reversed.
- A confounding variable affects both.
- The association occurs by chance.

Spurious correlations illustrate why an observed relationship should be investigated carefully before drawing conclusions.

**Key principle:** Correlation does not, by itself, establish causation.

## 11. Datasets Used in Course Examples

### Iris Dataset

Used to practice data loading, data inspection, category cleaning, descriptive statistics, grouping, and visualization.

### Customer Churn Dataset

Used to discuss customer characteristics, account information, revenue, satisfaction, customer lifetime value, and churn. The examples explore how customer outcomes may relate to tenure, payment methods, and other characteristics.

### Ames Housing Dataset

Used to demonstrate data quality checks, outlier investigation, variable selection, skewness handling, and feature transformation.

### Database Examples

SQLite examples demonstrate inspecting database tables and using SQL queries to retrieve, aggregate, and sort records before analysis in Pandas.

## 12. Key Takeaways

- Data quality assessment should precede reliable analysis.
- EDA combines descriptive statistics and visualization to understand data.
- Feature engineering and transformations can make variables more useful for modeling.
- Statistical inference requires careful interpretation of uncertainty and assumptions.
- Hypothesis testing should account for significance thresholds and multiple comparisons.
- Correlation should not be interpreted as proof of causation.
- Data preparation decisions should be justified by the dataset, analytical objective, and modeling method.

## Course Information

- **Course:** Exploratory Data Analysis for Machine Learning
- **Provider:** IBM
- **Platform:** Coursera

*These notes are a structured study summary based on the course materials. They are intended for learning and reference, not as a verbatim course transcript.*
