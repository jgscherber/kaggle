# Source: http://brettromero.com/wordpress/data-science-kaggle-walkthrough-cleaning-data/
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
import pandas as pd
import numpy as np

"""
o columns are largely categorical (11/16)
o will need to be transformed to into a better format
o timestamp in full number format

o .value_counts() on a series to get a per column distribution of the values
o good to look at over time too; NDF increaing year-over-year and country
  distributions also changing over time

o need to be careful not to over-generalize into the 2 major categories
  (US and NDF)

o Be careful when deleting rows with blank values ( >10% deleted )
o Need to be sure those rows don't continue unique information

df.describe(include='all') to info on all columns

"""

print("Reading files...")
df_train = pd.read_csv("train_users_2.csv", header=0,index_col=None)
df_test = pd.read_csv("test_users.csv", header=0, index_col=None)
df_all = pd.concat((df_train,df_test), axis=0, ignore_index=True)

print("Fixing timestamps...")
df_all["date_account_created"] = pd.to_datetime(df_all["date_account_created"]
                                                , format="%Y-%m-%d")
df_all["timestamp_first_active"] = pd.to_datetime(df_all["timestamp_first_active"]
                                                  , format="%Y%m%d%H%M%S")
# replaces blank entries (NaN) with the value in the timestamp_first_active
# column
df_all["date_account_created"].fillna(df_all.timestamp_first_active,
                                      inplace=True)
df_all.drop("date_first_booking", axis=1, inplace=True)


def remove_outliers(df, column, min_val, max_val):
    col_values = df[column].values
    # np.where(): takes (condition, x, y), returns x if true, y if false
    # np.logical_or(): compute truth value of x1 or x2 element wise
    df[column] = np.where(np.logical_or(
        col_values<=min_val, col_values>=max_val), np.NaN, col_values)
    return df
print("Fixing age column...")
df_all = remove_outliers(df_all, "age", 15, 75)
df_all['age'].fillna(-1,inplace=True)

print("Filling first_affiliate_tracked column...")
# replace blank rows with -1
df_all["first_affiliate_tracked"].fillna(-1, inplace=True)
# missing countries are from test data set
##print("Filling missing destinations...")
##df_all["country_destination"].fillna("NDF", inplace=True)
df_train = df_all.iloc[:213451]
df_test = df_all.iloc[213451:].reindex()




