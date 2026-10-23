# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md # Shop Performance — Machine Learning Data Preprocessing
# MAGIC
# MAGIC ## Dataset Introduction
# MAGIC
# MAGIC This practical uses the Shop Performance Orders dataset to demonstrate data preprocessing using Python, Pandas and scikit-learn.
# MAGIC
# MAGIC The original dataset contains 50,120 rows and 8 columns. It includes information such as OrderID, CustomerID, OrderDate, ProductID, Quantity, Discount, PaymentMethod and Status.
# MAGIC
# MAGIC The aim of this practical is to explore the dataset, identify and handle missing values, remove duplicates, encode categorical data, split the dataset into training and testing sets, and scale numerical features.
# MAGIC
# MAGIC These preprocessing steps help prepare the data for machine learning.
# MAGIC Part C Python Practical For Machine Learning Research Assgnment- i used the orders tables in our shop performance case study

# COMMAND ----------


# Import Pandas for working with data
import pandas as pd

# Load the Orders table from Databricks
orders_spark = spark.table("casestudy4.orders.orders")

# Convert the Spark DataFrame into a Pandas DataFrame
df = orders_spark.toPandas()

# Display the first five rows
display(df.head())


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data Exploration
# MAGIC
# MAGIC In this step, I explored the Shop Performance Orders dataset using Python and Pandas.
# MAGIC
# MAGIC I used `df.head()` to view the first five rows, `df.shape` to check the number of rows and columns, and `df.info()` to understand the column names and data types.
# MAGIC
# MAGIC I also used `df.isnull().sum()` to identify missing values and `df.duplicated().sum()` to check for duplicate records.
# MAGIC
# MAGIC The dataset originally contained 50,120 rows and 8 columns. I found 120 duplicate rows and missing values in OrderDate, Quantity, Discount and PaymentMethod.
# MAGIC
# MAGIC These checks helped me identify the data quality problems that needed to be addressed before preprocessing.

# COMMAND ----------


# Check the number of rows and columns in the Orders dataset
df.shape


# COMMAND ----------

# MAGIC %md
# MAGIC CHECKING THE COLUMN NAMES AND DATA TYPES

# COMMAND ----------


# Display the column names, data types and non-null values
df.info()



# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Cleaning
# MAGIC
# MAGIC In this step, I cleaned the Shop Performance Orders dataset to improve data quality before machine learning.
# MAGIC
# MAGIC I used `drop_duplicates()` to remove 120 duplicate rows and checked that no duplicate OrderIDs remained.
# MAGIC
# MAGIC I also handled missing values using the following methods:
# MAGIC
# MAGIC - **OrderDate:** Removed 35 rows with missing dates because the transaction dates could not be identified.
# MAGIC - **Quantity:** Replaced missing values with the median value of 2.
# MAGIC - **Discount:** Replaced missing values with 0, assuming no discount was applied.
# MAGIC - **PaymentMethod:** Replaced missing values with "Unknown" because the payment method was not recorded.
# MAGIC
# MAGIC After cleaning, I used `df.isnull().sum()` to confirm that no missing values remained.
# MAGIC
# MAGIC These steps helped prepare the dataset for machine learning.

# COMMAND ----------


# Check the number of missing values in each column
df.isnull().sum()


# COMMAND ----------

# MAGIC %md
# MAGIC CHECKING FOR DUPLICATES

# COMMAND ----------


# Count the number of duplicate rows in the Orders dataset
df.duplicated().sum()


# COMMAND ----------

# MAGIC %md
# MAGIC REMOVE DUPLICATE ROWS

# COMMAND ----------


# Remove duplicate rows from the Orders dataset
df = df.drop_duplicates().copy()

# Check whether any duplicate rows remain
df.duplicated().sum()



# COMMAND ----------

# MAGIC %md
# MAGIC Check for Duplicate Order IDs

# COMMAND ----------


# Check how many OrderID values are duplicated
df["OrderID"].duplicated().sum()


# COMMAND ----------

# MAGIC %md
# MAGIC Remove Missing Order Dates

# COMMAND ----------


# Check how many OrderDate values are missing
df["OrderDate"].isnull().sum()

# Remove rows where OrderDate is missing
df = df.dropna(subset=["OrderDate"]).copy()

# Check that no missing OrderDate values remain
df["OrderDate"].isnull().sum()


# COMMAND ----------

# MAGIC %md
# MAGIC Handle Missing Quantity Values

# COMMAND ----------


# Check how many Quantity values are missing
df["Quantity"].isnull().sum()

# Calculate the median quantity
quantity_median = df["Quantity"].median()

# Display the median
quantity_median


# COMMAND ----------

# MAGIC %md
# MAGIC Replace Missing Quantity Values

# COMMAND ----------


# Replace missing Quantity values with the median
df["Quantity"] = df["Quantity"].fillna(quantity_median)

# Check whether any missing Quantity values remain
df["Quantity"].isnull().sum()


# COMMAND ----------

# MAGIC %md
# MAGIC Handle Missing Discount Values-Missing values in the Discount column were replaced with 0 because we assumed that orders without recorded discounts had no discount applied.

# COMMAND ----------


# Replace missing Discount values with 0
# We assume missing discounts mean no discount was applied
df["Discount"] = df["Discount"].fillna(0)

# Check how many missing Discount values remain
df["Discount"].isnull().sum()


# COMMAND ----------

# MAGIC %md
# MAGIC Handle Missing PaymentMethod Values

# COMMAND ----------


# Replace missing PaymentMethod values with "Unknown"
df["PaymentMethod"] = df["PaymentMethod"].fillna("Unknown")

# Check whether any missing PaymentMethod values remain
df["PaymentMethod"].isnull().sum()


# COMMAND ----------

# MAGIC %md
# MAGIC Check for Remaining Missing Values

# COMMAND ----------


# Check for missing values in every column
df.isnull().sum()


# COMMAND ----------

# MAGIC %md
# MAGIC  Checking Our Categorical Data
# MAGIC inspecting the different payment methods in our dataset before encoding.

# COMMAND ----------


# Display the unique payment methods in the dataset
df["PaymentMethod"].unique()


# COMMAND ----------

# MAGIC %md # 3. Categorical Encoding
# MAGIC
# MAGIC In this step, I used One-Hot Encoding to convert the PaymentMethod column into numerical values.
# MAGIC
# MAGIC The PaymentMethod column contained five categories: Gateway, Wallet, CardToCard, Cash and Unknown.
# MAGIC
# MAGIC I used `pd.get_dummies()` to create a separate column for each payment method. The new columns contain 0s and 1s to show whether a particular payment method was used.
# MAGIC
# MAGIC One-Hot Encoding is suitable because payment methods do not have a natural order.
# MAGIC
# MAGIC This step helps prepare categorical data for machine learning algorithms that require numerical features.

# COMMAND ----------


# Convert PaymentMethod categories into numerical columns
# Each payment method gets its own column containing 0 or 1
df_encoded = pd.get_dummies(
    df,
    columns=["PaymentMethod"],
    dtype=int
)

# Display the first five rows of the encoded dataset
display(df_encoded.head())


# COMMAND ----------


# Display all column names after One-Hot Encoding
df_encoded.columns.tolist()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train/Test Split
# MAGIC
# MAGIC In this step, I divided the Shop Performance dataset into training and testing sets using the `train_test_split()` function from scikit-learn.
# MAGIC
# MAGIC I used 80% of the data for training and 20% for testing.
# MAGIC
# MAGIC The dataset contained 49,965 records after cleaning.
# MAGIC
# MAGIC - **Training data:** 39,972 records
# MAGIC - **Testing data:** 9,993 records
# MAGIC
# MAGIC I used `random_state=42` to make sure the data is split the same way each time the code runs.
# MAGIC
# MAGIC I also used `stratify=y` to maintain similar proportions of order statuses in both datasets.
# MAGIC
# MAGIC The purpose of splitting the data is to prepare a model for training and allow its performance to be evaluated on unseen data.
# MAGIC
# MAGIC To avoid data leakage, preprocessing methods that learn from data, such as median imputation and scaling, should be fitted using the training data only.

# COMMAND ----------


# Select the numerical and encoded categorical features
feature_columns = [
    "Quantity",
    "Discount",
    "PaymentMethod_CardToCard",
    "PaymentMethod_Cash",
    "PaymentMethod_Gateway",
    "PaymentMethod_Unknown",
    "PaymentMethod_Wallet"
]

# X contains the features used for machine learning
X = df_encoded[feature_columns]

# y contains the target we want to predict
y = df_encoded["Status"]

# Check the dimensions of the features and target
X.shape, y.shape


# COMMAND ----------

# MAGIC %md
# MAGIC Split the Dataset into Training and Testing Sets

# COMMAND ----------


# Import the function used to split datasets
from sklearn.model_selection import train_test_split

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,       # Use 20% of the data for testing
    random_state=42,     # Make the split reproducible
    stratify=y           # Keep similar Status proportions in both sets
)

# Display the number of rows and columns in each dataset
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# COMMAND ----------

# MAGIC %md ## 5. Feature Scaling
# MAGIC
# MAGIC In this step, I used StandardScaler from scikit-learn to scale two numerical columns: Quantity and Discount.
# MAGIC
# MAGIC Feature scaling helps put numerical features on comparable scales so that features with larger values do not unfairly influence certain machine learning algorithms.
# MAGIC
# MAGIC I used `fit_transform()` on the training data to calculate the mean and standard deviation and apply scaling.
# MAGIC
# MAGIC I then used `transform()` on the testing data to apply the same scaling values learned from the training data.
# MAGIC
# MAGIC After scaling, the training features had a mean of approximately 0 and a standard deviation of approximately 1.
# MAGIC
# MAGIC This approach helps prevent data leakage because the scaler does not learn information from the testing dataset.
# MAGIC

# COMMAND ----------


# Import StandardScaler for numerical feature scaling
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Create copies to keep the original datasets unchanged
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Select the two numerical columns to scale
numeric_columns = ["Quantity", "Discount"]

# Fit the scaler on training data and scale both columns
X_train_scaled[numeric_columns] = scaler.fit_transform(
    X_train[numeric_columns]
)

# Use the same scaler to transform the testing data
X_test_scaled[numeric_columns] = scaler.transform(
    X_test[numeric_columns]
)

# Display the first five rows of the scaled training data
display(X_train_scaled.head())


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Final Preprocessing — Preventing Data Leakage
# MAGIC
# MAGIC In this section, I use a fresh copy of the original Orders dataset and prepare the training and testing data correctly.
# MAGIC
# MAGIC I split the dataset before calculating replacement values for missing numerical data. This ensures that information from the testing dataset is not used when preparing the training data.

# COMMAND ----------


# Reload the original Orders dataset
# This ensures we start from the raw data again
df_final = spark.table("casestudy4.orders.orders").toPandas()

# Remove duplicate rows
df_final = df_final.drop_duplicates().copy()

# Remove records with missing OrderDate values
df_final = df_final.dropna(subset=["OrderDate"]).copy()

# Display the dataset dimensions
df_final.shape


# COMMAND ----------

# MAGIC %md
# MAGIC Split the Original Data Before Imputation

# COMMAND ----------


# Import the function used to split the dataset
from sklearn.model_selection import train_test_split

# Select the features for machine learning
# These still contain missing values at this stage
X_final = df_final[["Quantity", "Discount", "PaymentMethod"]]

# Select the target column
y_final = df_final["Status"]

# Split the data into 80% training and 20% testing
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X_final,
    y_final,
    test_size=0.2,
    random_state=42,
    stratify=y_final
)

# Display the dimensions of the training and testing datasets
X_train_final.shape, X_test_final.shape


# COMMAND ----------

# MAGIC %md
# MAGIC Handle Missing Quantity Values Correctly

# COMMAND ----------


# Import SimpleImputer to handle missing values
from sklearn.impute import SimpleImputer

# Create an imputer that uses the median
quantity_imputer = SimpleImputer(strategy="median")

# Create copies of the training and testing datasets
X_train_final = X_train_final.copy()
X_test_final = X_test_final.copy()

# Calculate the median from training data and fill missing values
X_train_final[["Quantity"]] = quantity_imputer.fit_transform(
    X_train_final[["Quantity"]]
)

# Apply the training median to missing values in testing data
X_test_final[["Quantity"]] = quantity_imputer.transform(
    X_test_final[["Quantity"]]
)

# Check remaining missing Quantity values
X_train_final["Quantity"].isnull().sum(), X_test_final["Quantity"].isnull().sum()


# COMMAND ----------

# MAGIC %md
# MAGIC Handle Missing Discount Values

# COMMAND ----------


# Replace missing Discount values with 0 in training data
X_train_final["Discount"] = X_train_final["Discount"].fillna(0)

# Replace missing Discount values with 0 in testing data
X_test_final["Discount"] = X_test_final["Discount"].fillna(0)

# Check missing Discount values in both datasets
(
    X_train_final["Discount"].isnull().sum(),
    X_test_final["Discount"].isnull().sum()
)


# COMMAND ----------

# MAGIC %md
# MAGIC Handle Missing PaymentMethod Values-replacing missing payment methods with "Unknown" in both the training and testing datasets.

# COMMAND ----------


# Replace missing PaymentMethod values in the training dataset
X_train_final["PaymentMethod"] = X_train_final["PaymentMethod"].fillna("Unknown")

# Replace missing PaymentMethod values in the testing dataset
X_test_final["PaymentMethod"] = X_test_final["PaymentMethod"].fillna("Unknown")

# Check that no missing PaymentMethod values remain
(
    X_train_final["PaymentMethod"].isnull().sum(),
    X_test_final["PaymentMethod"].isnull().sum()
)


# COMMAND ----------

# MAGIC %md
# MAGIC One-Hot Encoding Without Data Leakage
# MAGIC  use scikit-learn's OneHotEncoder so the encoder learns the categories from the training data

# COMMAND ----------


# Import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# Create the encoder
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Learn payment method categories from training data
train_encoded = encoder.fit_transform(
    X_train_final[["PaymentMethod"]]
)

# Apply the same encoding to testing data
test_encoded = encoder.transform(
    X_test_final[["PaymentMethod"]]
)

# Display the names of the encoded columns
encoder.get_feature_names_out(["PaymentMethod"])


# COMMAND ----------

# MAGIC %md
# MAGIC Combining the Encoded Columns with Numerical Features- combine the encoded payment methods with Quantity and Discount.

# COMMAND ----------


# Get the names of the encoded payment method columns
encoded_columns = encoder.get_feature_names_out(["PaymentMethod"])

# Convert the encoded training data into a Pandas DataFrame
train_encoded_df = pd.DataFrame(
    train_encoded,
    columns=encoded_columns,
    index=X_train_final.index
)

# Convert the encoded testing data into a Pandas DataFrame
test_encoded_df = pd.DataFrame(
    test_encoded,
    columns=encoded_columns,
    index=X_test_final.index
)

# Combine numerical and encoded columns for training
X_train_ready = pd.concat(
    [X_train_final[["Quantity", "Discount"]], train_encoded_df],
    axis=1
)

# Combine numerical and encoded columns for testing
X_test_ready = pd.concat(
    [X_test_final[["Quantity", "Discount"]], test_encoded_df],
    axis=1
)

# Check the dimensions of both datasets
X_train_ready.shape, X_test_ready.shape


# COMMAND ----------

# MAGIC %md Final Feature Scaling- scale Quantity and Discount using StandardScaler.

# COMMAND ----------


# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create the scaler
final_scaler = StandardScaler()

# Create copies of the datasets
X_train_ready = X_train_ready.copy()
X_test_ready = X_test_ready.copy()

# Select the two numerical features
numeric_columns = ["Quantity", "Discount"]

# Fit the scaler on training data only
X_train_ready[numeric_columns] = final_scaler.fit_transform(
    X_train_ready[numeric_columns]
)

# Apply the same scaling to testing data
X_test_ready[numeric_columns] = final_scaler.transform(
    X_test_ready[numeric_columns]
)

# Check the mean of the scaled training features
X_train_ready[numeric_columns].mean().round(2)


# COMMAND ----------

# MAGIC %md
# MAGIC Data Quality Check

# COMMAND ----------


# Check missing values in the final training dataset
X_train_ready.isnull().sum()

# Check missing values in the final testing dataset
X_test_ready.isnull().sum()

# Display the final dataset dimensions and missing-value counts
print("Training shape:", X_train_ready.shape)
print("Testing shape:", X_test_ready.shape)
print("Training missing values:", X_train_ready.isnull().sum().sum())
print("Testing missing values:", X_test_ready.isnull().sum().sum())


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Conclusion — Machine Learning Research Assignment
# MAGIC
# MAGIC In this Machine Learning Research Assignment, I used Python, Pandas and scikit-learn to demonstrate the importance of data preparation and preprocessing.
# MAGIC
# MAGIC For the practical section, I used the Shop Performance Orders dataset. I explored the data, checked for missing values and duplicates, and applied appropriate data cleaning methods.
# MAGIC
# MAGIC I used One-Hot Encoding to convert categorical data into numerical features and StandardScaler to scale the Quantity and Discount columns.
# MAGIC
# MAGIC I also divided the dataset into 80% training data and 20% testing data. To prevent data leakage, I fitted the imputer, encoder and scaler using the training data only.
# MAGIC
# MAGIC The final training dataset contained 39,972 rows and 7 features, while the testing dataset contained 9,993 rows and 7 features. Both datasets had no missing values.
# MAGIC
# MAGIC Through this practical, I gained a better understanding of how data cleaning, encoding, feature scaling and data splitting help prepare datasets for machine learning.
# MAGIC
# MAGIC Overall, this assignment helped me improve my understanding of machine learning concepts and the importance of preparing quality data before building a model.