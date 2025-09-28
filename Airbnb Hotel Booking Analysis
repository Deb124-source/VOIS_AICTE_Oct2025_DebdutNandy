# Airbnb Hotel Booking Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset
df = pd.read_csv('/content/drive/MyDrive/1730285881-Airbnb_Open_Data(in) (3) (1).csv')

# Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')

# Data Exploration
print("Dataset Info:")
print(df.info())
print("\nDuplicate Records:")
print(df.duplicated().value_counts())
print("\nFirst 5 rows:")
print(df.head())

# Data Cleaning
# Drop duplicate records
df.drop_duplicates(inplace=True)

# Drop columns with insufficient data
df.drop(['house_rules','license'], axis=1, inplace=True)

# Remove dollar signs and commas from price and service fee columns
df['price'] = df['price'].str.replace('$', '', regex=False)
df['service fee'] = df['service fee'].str.replace('$', '', regex=False)
df['price'] = df['price'].str.replace(',', '', regex=False)
df['service fee'] = df['service fee'].str.replace(',', '', regex=False)

# Rename columns to include dollar sign
df.rename(columns={'price': '$price', 'service fee': '$service fee'}, inplace=True)

# Drop all records with missing values
df.dropna(inplace=True)

# Change data types
df['$price'] = df['$price'].astype(float)
df['$service fee'] = df['$service fee'].astype(float)
df['id'] = df['id'].astype(int)
df['host id'] = df['host id'].astype(int)
df['Construction year'] = df['Construction year'].astype(int)

# Correct spelling of 'brooklyn'
df.loc[df['neighbourhood group'] == 'brookln', 'neighbourhood group'] = 'brooklyn'

# Remove outliers in availability 365 column
df = df.drop(df[df['availability 365'] > 500].index)

print("\nAfter cleaning - Duplicate Records:")
print(df.duplicated().value_counts())
print("\nAfter cleaning - Dataset Info:")
print(df.info())

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Exploratory Data Analysis

# 1. What are the different property types in the Dataset?
property_types = df['room type'].value_counts().to_frame()
print("\nProperty Types:")
print(property_types)

# Visualization for property types
plt.figure(figsize=(10, 6))
room_type_bar = plt.bar(property_types.index, property_types['count'])
plt.bar_label(room_type_bar, labels=property_types.loc[:, 'count'], padding=4)
plt.ylim([0, 50000])
plt.xlabel('Room Type')
plt.ylabel('Room Type Count')
plt.title('Property types and their count in the dataset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Which neighborhoods group has the highest number of listings?
hood_group = df['neighbourhood group'].value_counts().to_frame()
print("\nNeighbourhood Groups:")
print(hood_group)

# Visualization for neighbourhood groups
plt.figure(figsize=(10, 6))
hood_group_bar = plt.bar(hood_group.index, hood_group.loc[:, 'count'])
plt.bar_label(hood_group_bar, labels=hood_group.loc[:, 'count'], padding=4)
plt.ylim([0, 40000])
plt.xlabel('Neighbourhood Group')
plt.ylabel('Numbers of listings')
plt.xticks(rotation=45)
plt.title('Which Neighbourhood Group has the highest number of listings')
plt.tight_layout()
plt.show()

# 3. Which neighborhood has the highest number of entire home/apartment type listings?
entire_home_listings = df[df['room type'] == 'Entire home/apt']
entire_home_by_neighbourhood = entire_home_listings['neighbourhood group'].value_counts()
print("\nEntire home/apt listings by neighbourhood group:")
print(entire_home_by_neighbourhood)

# 4. Is there a relationship between availability and price of property?
plt.figure(figsize=(10, 6))
plt.scatter(df['availability 365'], df['$price'], alpha=0.5)
plt.xlabel('Availability (days)')
plt.ylabel('Price ($)')
plt.title('Relationship between Availability and Price')
plt.tight_layout()
plt.show()

# Correlation between availability and price
correlation_avail_price = df['availability 365'].corr(df['$price'])
print(f"\nCorrelation between availability and price: {correlation_avail_price:.2f}")

# 5. Which is the busiest host with the highest number of listings?
busiest_hosts = df.groupby('host id')['calculated host listings count'].max().sort_values(ascending=False).head(10)
print("\nTop 10 hosts with highest number of listings:")
print(busiest_hosts)

# 6. Which neighborhood has the highest number of private room type listings?
private_room_listings = df[df['room type'] == 'Private room']
private_room_by_neighbourhood = private_room_listings['neighbourhood group'].value_counts()
print("\nPrivate room listings by neighbourhood group:")
print(private_room_by_neighbourhood)

# 7. Which are the top 10 most reviewed listings?
top_reviewed = df.nlargest(10, 'number of reviews')[['NAME', 'number of reviews', 'neighbourhood group']]
print("\nTop 10 most reviewed listings:")
print(top_reviewed)

# 8. Is there a correlation between host service and listings reviews?
# Using service fee and number of reviews as proxies
correlation_service_reviews = df['$service fee'].corr(df['number of reviews'])
print(f"\nCorrelation between service fee and number of reviews: {correlation_service_reviews:.2f}")

# 9. Is there a correlation between the location of a listing and its service fee?
# Using neighbourhood group and service fee
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='neighbourhood group', y='$service fee')
plt.xticks(rotation=45)
plt.title('Service Fee by Neighbourhood Group')
plt.tight_layout()
plt.show()

# Average service fee by neighbourhood group
avg_service_fee_by_hood = df.groupby('neighbourhood group')['$service fee'].mean().sort_values(ascending=False)
print("\nAverage service fee by neighbourhood group:")
print(avg_service_fee_by_hood)

# Additional Analysis

# Price distribution by room type
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='room type', y='$price')
plt.xticks(rotation=45)
plt.title('Price Distribution by Room Type')
plt.tight_layout()
plt.show()

# Review rate distribution
plt.figure(figsize=(10, 6))
plt.hist(df['review rate number'], bins=20, edgecolor='black')
plt.xlabel('Review Rate')
plt.ylabel('Frequency')
plt.title('Distribution of Review Rates')
plt.tight_layout()
plt.show()

# Availability by neighbourhood group
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='neighbourhood group', y='availability 365')
plt.xticks(rotation=45)
plt.title('Availability by Neighbourhood Group')
plt.tight_layout()
plt.show()

print("\nAnalysis completed successfully!")
