# Source: http://brettromero.com/wordpress/data-science-kaggle-walkthrough-cleaning-data/

import pandas as pd

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
# replaces blank columns (NaN) with the value in the timestamp_first_active
# column
df_all["date_account_created"].fillna(df_all.timestamp_first_active,
                                      inplace=True)
df_all.drop("date_first_booking", axis=1, inplace=True)






