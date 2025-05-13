# pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
# For this example, we'll create a sample sales dataset
# In a real scenario, you'd use: df = pd.read_csv('your_dataset.csv')

# Create a sample sales dataset
np.random.seed(42)  # For reproducibility

# Generate random data
n_rows = 1000
customer_ids = [f'CUST{i:04d}' for i in range(1, n_rows + 1)]
products = ['Laptop', 'Smartphone', 'Tablet', 'Monitor', 'Headphones']
regions = ['North', 'South', 'East', 'West', 'Central']
payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Cash', 'Bank Transfer']

# Create a DataFrame
data = {
    'customer_id': np.random.choice(customer_ids, n_rows),
    'date': pd.date_range(start='2023-01-01', periods=n_rows, freq='H'),
    'product': np.random.choice(products, n_rows),
    'quantity': np.random.randint(1, 10, n_rows),
    'unit_price': np.random.uniform(10, 1000, n_rows).round(2),
    'region': np.random.choice(regions, n_rows),
    'payment_method': np.random.choice(payment_methods, n_rows)
}

# Calculate total_price
data['total_price'] = (data['quantity'] * data['unit_price']).round(2)

# Introduce some missing values (about 5% of data)
for col in ['product', 'quantity', 'region', 'payment_method']:
    mask = np.random.random(n_rows) < 0.05
    data[col] = np.where(mask, np.nan, data[col])

# Create DataFrame
df = pd.DataFrame(data)

# Step 2: Display the first few rows of the dataset
print("STEP 2: First 5 rows of the dataset")
print(df.head())
print("\n" + "="*80 + "\n")

# Step 3: Explore the structure of the dataset
print("STEP 3: Dataset Information")
print("\nDataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nSummary Statistics:")
print(df.describe())
print("\nMissing Values Count:")
print(df.isnull().sum())
print("\n" + "="*80 + "\n")

# Step 4: Clean the dataset
print("STEP 4: Cleaning the Dataset")

# Store the original shape for comparison
original_shape = df.shape

# Create a copy of the original data for comparison
df_cleaned = df.copy()

# Method 1: Fill missing values
print("\nMethod 1: Filling Missing Values")
# Fill missing product values with 'Unknown'
df_cleaned['product'] = df_cleaned['product'].fillna('Unknown')
# Fill missing quantity values with the median quantity
df_cleaned['quantity'] = df_cleaned['quantity'].fillna(df_cleaned['quantity'].median())
# Fill missing region values with the most common region
df_cleaned['region'] = df_cleaned['region'].fillna(df_cleaned['region'].mode()[0])
# Fill missing payment_method values with the most common method
df_cleaned['payment_method'] = df_cleaned['payment_method'].fillna(df_cleaned['payment_method'].mode()[0])

print("Missing Values After Filling:")
print(df_cleaned.isnull().sum())

# Method 2: Create another copy and demonstrate dropping rows with missing values
df_dropped = df.copy()
print("\nMethod 2: Dropping Rows with Missing Values")
print(f"Original shape: {df_dropped.shape}")
df_dropped = df_dropped.dropna()
print(f"Shape after dropping rows with missing values: {df_dropped.shape}")
print(f"Rows removed: {original_shape[0] - df_dropped.shape[0]}")

# Display cleaned data
print("\nCleaned Dataset (First 5 rows):")
print(df_cleaned.head())
print("\n" + "="*80 + "\n")

# Step 5: Visualize some aspects of the cleaned data
print("STEP 5: Basic Data Visualization")

# Visualization 1: Sales by product
plt.figure(figsize=(10, 6))
product_sales = df_cleaned.groupby('product')['total_price'].sum().sort_values(ascending=False)
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.title('Total Sales by Product')
plt.xlabel('Product')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sales_by_product.png')

# Visualization 2: Sales by region
plt.figure(figsize=(10, 6))
region_sales = df_cleaned.groupby('region')['total_price'].sum().sort_values(ascending=False)
sns.barplot(x=region_sales.index, y=region_sales.values)
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales ($)')
plt.tight_layout()
plt.savefig('sales_by_region.png')

print("Basic visualizations have been created and saved as PNG files")
print("\nData analysis and cleaning completed successfully!")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
# For this example, we'll create a sample sales dataset
# In a real scenario, you'd use: df = pd.read_csv('your_dataset.csv')

# Create a sample sales dataset
np.random.seed(42)  # For reproducibility

# Generate random data
n_rows = 1000
customer_ids = [f'CUST{i:04d}' for i in range(1, n_rows + 1)]
products = ['Laptop', 'Smartphone', 'Tablet', 'Monitor', 'Headphones']
regions = ['North', 'South', 'East', 'West', 'Central']
payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Cash', 'Bank Transfer']
customer_types = ['Individual', 'Business', 'Education', 'Government']

# Create a DataFrame
data = {
    'customer_id': np.random.choice(customer_ids, n_rows),
    'date': pd.date_range(start='2023-01-01', periods=n_rows, freq='H'),
    'product': np.random.choice(products, n_rows),
    'quantity': np.random.randint(1, 10, n_rows),
    'unit_price': np.random.uniform(10, 1000, n_rows).round(2),
    'region': np.random.choice(regions, n_rows),
    'payment_method': np.random.choice(payment_methods, n_rows),
    'customer_type': np.random.choice(customer_types, n_rows),
    'discount_percent': np.random.choice([0, 5, 10, 15, 20], n_rows)
}

# Calculate total_price (with discount applied)
data['total_price'] = (data['quantity'] * data['unit_price'] * (1 - data['discount_percent']/100)).round(2)

# Introduce some missing values (about 5% of data)
for col in ['product', 'quantity', 'region', 'payment_method']:
    mask = np.random.random(n_rows) < 0.05
    data[col] = np.where(mask, np.nan, data[col])

# Create DataFrame
df = pd.DataFrame(data)

# Step 2: Display the first few rows of the dataset
print("STEP 2: First 5 rows of the dataset")
print(df.head())
print("\n" + "="*80 + "\n")

# Step 3: Explore the structure of the dataset
print("STEP 3: Dataset Information")
print("\nDataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values Count:")
print(df.isnull().sum())
print("\n" + "="*80 + "\n")

# Step 4: Clean the dataset
print("STEP 4: Cleaning the Dataset")

# Store the original shape for comparison
original_shape = df.shape

# Create a copy of the original data for comparison
df_cleaned = df.copy()

# Method 1: Fill missing values
print("\nMethod 1: Filling Missing Values")
# Fill missing product values with 'Unknown'
df_cleaned['product'] = df_cleaned['product'].fillna('Unknown')
# Fill missing quantity values with the median quantity
df_cleaned['quantity'] = df_cleaned['quantity'].fillna(df_cleaned['quantity'].median())
# Fill missing region values with the most common region
df_cleaned['region'] = df_cleaned['region'].fillna(df_cleaned['region'].mode()[0])
# Fill missing payment_method values with the most common method
df_cleaned['payment_method'] = df_cleaned['payment_method'].fillna(df_cleaned['payment_method'].mode()[0])

print("Missing Values After Filling:")
print(df_cleaned.isnull().sum())

# Method 2: Create another copy and demonstrate dropping rows with missing values
df_dropped = df.copy()
print("\nMethod 2: Dropping Rows with Missing Values")
print(f"Original shape: {df_dropped.shape}")
df_dropped = df_dropped.dropna()
print(f"Shape after dropping rows with missing values: {df_dropped.shape}")
print(f"Rows removed: {original_shape[0] - df_dropped.shape[0]}")

# Display cleaned data
print("\nCleaned Dataset (First 5 rows):")
print(df_cleaned.head())
print("\n" + "="*80 + "\n")

# Step 5: Compute basic statistics on numerical columns
print("STEP 5: Basic Statistics of Numerical Columns")
numerical_stats = df_cleaned.describe()
print(numerical_stats)
print("\n")

# Additional statistics not included in describe()
print("Additional Statistics:")
numerical_columns = ['quantity', 'unit_price', 'total_price', 'discount_percent']
for col in numerical_columns:
    print(f"\n{col} Statistics:")
    print(f"Median: {df_cleaned[col].median()}")
    print(f"Mode: {df_cleaned[col].mode()[0]}")
    print(f"Variance: {df_cleaned[col].var()}")
    print(f"Skewness: {df_cleaned[col].skew()}")
    print(f"Kurtosis: {df_cleaned[col].kurt()}")
print("\n" + "="*80 + "\n")

# Step 6: Perform groupings on categorical columns
print("STEP 6: Grouping Analysis")

# Group by product
print("\nSales Statistics by Product:")
product_stats = df_cleaned.groupby('product').agg({
    'quantity': ['count', 'sum', 'mean', 'median'],
    'unit_price': ['mean', 'median', 'min', 'max'],
    'total_price': ['sum', 'mean', 'median']
})
print(product_stats)

# Group by region
print("\nSales Statistics by Region:")
region_stats = df_cleaned.groupby('region').agg({
    'quantity': ['count', 'sum', 'mean'],
    'total_price': ['sum', 'mean', 'median']
})
print(region_stats)

# Group by customer type
print("\nSales Statistics by Customer Type:")
customer_type_stats = df_cleaned.groupby('customer_type').agg({
    'quantity': ['count', 'sum', 'mean'],
    'unit_price': ['mean', 'median'],
    'discount_percent': ['mean', 'median'],
    'total_price': ['sum', 'mean', 'median']
})
print(customer_type_stats)

# Cross-tabulation of product and region
print("\nCross-tabulation of Products by Region (Count):")
product_region_crosstab = pd.crosstab(df_cleaned['product'], df_cleaned['region'])
print(product_region_crosstab)

# Average purchase amount by product and customer type
print("\nAverage Purchase Amount by Product and Customer Type:")
avg_by_product_customer = df_cleaned.pivot_table(
    values='total_price',
    index='product',
    columns='customer_type',
    aggfunc='mean'
)
print(avg_by_product_customer)
print("\n" + "="*80 + "\n")

# Step 7: Identify patterns and interesting findings
print("STEP 7: Pattern Identification and Interesting Findings")

# Find the most popular product by region
most_popular_by_region = df_cleaned.groupby(['region', 'product']).size().reset_index(name='count')
most_popular_by_region = most_popular_by_region.sort_values(['region', 'count'], ascending=[True, False])
most_popular_product = most_popular_by_region.groupby('region').first()
print("\nMost Popular Product by Region:")
print(most_popular_product[['product', 'count']])

# Analyze discount patterns
print("\nDiscount Analysis by Customer Type:")
discount_analysis = df_cleaned.groupby('customer_type')['discount_percent'].value_counts().unstack().fillna(0)
print(discount_analysis)

# Find high-value purchases (top 5%)
high_value_threshold = df_cleaned['total_price'].quantile(0.95)
high_value_purchases = df_cleaned[df_cleaned['total_price'] >= high_value_threshold]
print(f"\nHigh-Value Purchases (Top 5%, Threshold: ${high_value_threshold:.2f}):")
high_value_stats = high_value_purchases.groupby('product').agg({
    'total_price': ['count', 'sum', 'mean']
})
print(high_value_stats)

# Payment method preferences by customer type
payment_preferences = pd.crosstab(
    df_cleaned['customer_type'], 
    df_cleaned['payment_method'], 
    normalize='index'
).round(2)
print("\nPayment Method Preferences by Customer Type (as proportion):")
print(payment_preferences)

# Sales trends by hour of day
df_cleaned['hour'] = df_cleaned['date'].dt.hour
hourly_sales = df_cleaned.groupby('hour')['total_price'].sum()
peak_hour = hourly_sales.idxmax()
print(f"\nPeak Sales Hour: {peak_hour}:00 with ${hourly_sales[peak_hour]:.2f} in sales")
print("\n" + "="*80 + "\n")

# Step 8: Visualize some aspects of the cleaned data and insights
print("STEP 8: Data Visualization")

# Set a consistent style for all plots
sns.set(style="whitegrid")

# Visualization 1: Sales by product
plt.figure(figsize=(10, 6))
product_sales = df_cleaned.groupby('product')['total_price'].sum().sort_values(ascending=False)
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.title('Total Sales by Product')
plt.xlabel('Product')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sales_by_product.png')

# Visualization 2: Sales by region
plt.figure(figsize=(10, 6))
region_sales = df_cleaned.groupby('region')['total_price'].sum().sort_values(ascending=False)
sns.barplot(x=region_sales.index, y=region_sales.values)
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales ($)')
plt.tight_layout()
plt.savefig('sales_by_region.png')

# Visualization 3: Average purchase by customer type
plt.figure(figsize=(10, 6))
customer_avg_purchase = df_cleaned.groupby('customer_type')['total_price'].mean().sort_values(ascending=False)
sns.barplot(x=customer_avg_purchase.index, y=customer_avg_purchase.values)
plt.title('Average Purchase Amount by Customer Type')
plt.xlabel('Customer Type')
plt.ylabel('Average Purchase ($)')
plt.tight_layout()
plt.savefig('avg_purchase_by_customer_type.png')

# Visualization 4: Distribution of purchase amounts
plt.figure(figsize=(12, 6))
sns.histplot(data=df_cleaned, x='total_price', kde=True, bins=30)
plt.title('Distribution of Purchase Amounts')
plt.xlabel('Purchase Amount ($)')
plt.ylabel('Frequency')
plt.axvline(x=df_cleaned['total_price'].mean(), color='red', linestyle='--', label=f'Mean: ${df_cleaned["total_price"].mean():.2f}')
plt.axvline(x=df_cleaned['total_price'].median(), color='green', linestyle='--', label=f'Median: ${df_cleaned["total_price"].median():.2f}')
plt.legend()
plt.tight_layout()
plt.savefig('purchase_amount_distribution.png')

# Visualization 5: Hourly sales pattern
plt.figure(figsize=(12, 6))
sns.lineplot(x=hourly_sales.index, y=hourly_sales.values)
plt.title('Sales by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Total Sales ($)')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig('hourly_sales_pattern.png')

# Visualization 6: Heatmap of product sales by region
plt.figure(figsize=(12, 8))
product_region_sales = df_cleaned.pivot_table(
    values='total_price', 
    index='product', 
    columns='region', 
    aggfunc='sum'
)
sns.heatmap(product_region_sales, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('Heatmap of Product Sales by Region')
plt.tight_layout()
plt.savefig('product_region_heatmap.png')

print("Visualizations have been created and saved as PNG files")
print("\nData analysis and insights extraction completed successfully!")

# Step 9: Create Focused Visualizations (Four Different Types)
print("STEP 9: Four Different Types of Visualizations")

# Set a consistent style and color palette for all plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
color_palette = sns.color_palette("viridis", 5)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_palette)

# 1. LINE CHART: Monthly Sales Trend
# First, create monthly aggregates for time series analysis
df_cleaned['month'] = df_cleaned['date'].dt.to_period('M')
monthly_sales = df_cleaned.groupby('month')['total_price'].sum()
monthly_sales.index = monthly_sales.index.astype(str)  # Convert period to string for plotting

plt.figure()
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, markersize=8)
plt.title('Monthly Sales Trend (2023)', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Total Sales ($)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('monthly_sales_trend.png')
print("1. Line Chart: Monthly Sales Trend created")

# 2. BAR CHART: Average Unit Price by Product Category
avg_price_by_product = df_cleaned.groupby('product')['unit_price'].mean().sort_values(ascending=False)

plt.figure()
bars = plt.bar(avg_price_by_product.index, avg_price_by_product.values, color=color_palette)
plt.title('Average Unit Price by Product Category', fontsize=16, fontweight='bold')
plt.xlabel('Product Category', fontsize=14)
plt.ylabel('Average Price ($)', fontsize=14)
plt.xticks(rotation=45)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'${height:.2f}', ha='center', fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('avg_price_by_product.png')
print("2. Bar Chart: Average Unit Price by Product created")

# 3. HISTOGRAM: Distribution of Total Prices with Normal Curve
plt.figure()
sns.histplot(df_cleaned['total_price'], kde=True, bins=30, color=color_palette[2])

# Add statistical markers
mean_price = dfimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
# For this example, we'll create a sample sales dataset
# In a real scenario, you'd use: df = pd.read_csv('your_dataset.csv')

# Create a sample sales dataset
np.random.seed(42)  # For reproducibility

# Generate random data
n_rows = 1000
customer_ids = [f'CUST{i:04d}' for i in range(1, n_rows + 1)]
products = ['Laptop', 'Smartphone', 'Tablet', 'Monitor', 'Headphones']
regions = ['North', 'South', 'East', 'West', 'Central']
payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Cash', 'Bank Transfer']
customer_types = ['Individual', 'Business', 'Education', 'Government']

# Create a DataFrame
data = {
    'customer_id': np.random.choice(customer_ids, n_rows),
    'date': pd.date_range(start='2023-01-01', periods=n_rows, freq='H'),
    'product': np.random.choice(products, n_rows),
    'quantity': np.random.randint(1, 10, n_rows),
    'unit_price': np.random.uniform(10, 1000, n_rows).round(2),
    'region': np.random.choice(regions, n_rows),
    'payment_method': np.random.choice(payment_methods, n_rows),
    'customer_type': np.random.choice(customer_types, n_rows),
    'discount_percent': np.random.choice([0, 5, 10, 15, 20], n_rows)
}

# Calculate total_price (with discount applied)
data['total_price'] = (data['quantity'] * data['unit_price'] * (1 - data['discount_percent']/100)).round(2)

# Introduce some missing values (about 5% of data)
for col in ['product', 'quantity', 'region', 'payment_method']:
    mask = np.random.random(n_rows) < 0.05
    data[col] = np.where(mask, np.nan, data[col])

# Create DataFrame
df = pd.DataFrame(data)

# Step 2: Display the first few rows of the dataset
print("STEP 2: First 5 rows of the dataset")
print(df.head())
print("\n" + "="*80 + "\n")

# Step 3: Explore the structure of the dataset
print("STEP 3: Dataset Information")
print("\nDataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values Count:")
print(df.isnull().sum())
print("\n" + "="*80 + "\n")

# Step 4: Clean the dataset
print("STEP 4: Cleaning the Dataset")

# Store the original shape for comparison
original_shape = df.shape

# Create a copy of the original data for comparison
df_cleaned = df.copy()

# Method 1: Fill missing values
print("\nMethod 1: Filling Missing Values")
# Fill missing product values with 'Unknown'
df_cleaned['product'] = df_cleaned['product'].fillna('Unknown')
# Fill missing quantity values with the median quantity
df_cleaned['quantity'] = df_cleaned['quantity'].fillna(df_cleaned['quantity'].median())
# Fill missing region values with the most common region
df_cleaned['region'] = df_cleaned['region'].fillna(df_cleaned['region'].mode()[0])
# Fill missing payment_method values with the most common method
df_cleaned['payment_method'] = df_cleaned['payment_method'].fillna(df_cleaned['payment_method'].mode()[0])

print("Missing Values After Filling:")
print(df_cleaned.isnull().sum())

# Method 2: Create another copy and demonstrate dropping rows with missing values
df_dropped = df.copy()
print("\nMethod 2: Dropping Rows with Missing Values")
print(f"Original shape: {df_dropped.shape}")
df_dropped = df_dropped.dropna()
print(f"Shape after dropping rows with missing values: {df_dropped.shape}")
print(f"Rows removed: {original_shape[0] - df_dropped.shape[0]}")

# Display cleaned data
print("\nCleaned Dataset (First 5 rows):")
print(df_cleaned.head())
print("\n" + "="*80 + "\n")

# Step 5: Compute basic statistics on numerical columns
print("STEP 5: Basic Statistics of Numerical Columns")
numerical_stats = df_cleaned.describe()
print(numerical_stats)
print("\n")

# Additional statistics not included in describe()
print("Additional Statistics:")
numerical_columns = ['quantity', 'unit_price', 'total_price', 'discount_percent']
for col in numerical_columns:
    print(f"\n{col} Statistics:")
    print(f"Median: {df_cleaned[col].median()}")
    print(f"Mode: {df_cleaned[col].mode()[0]}")
    print(f"Variance: {df_cleaned[col].var()}")
    print(f"Skewness: {df_cleaned[col].skew()}")
    print(f"Kurtosis: {df_cleaned[col].kurt()}")
print("\n" + "="*80 + "\n")

# Step 6: Perform groupings on categorical columns
print("STEP 6: Grouping Analysis")

# Group by product
print("\nSales Statistics by Product:")
product_stats = df_cleaned.groupby('product').agg({
    'quantity': ['count', 'sum', 'mean', 'median'],
    'unit_price': ['mean', 'median', 'min', 'max'],
    'total_price': ['sum', 'mean', 'median']
})
print(product_stats)

# Group by region
print("\nSales Statistics by Region:")
region_stats = df_cleaned.groupby('region').agg({
    'quantity': ['count', 'sum', 'mean'],
    'total_price': ['sum', 'mean', 'median']
})
print(region_stats)

# Group by customer type
print("\nSales Statistics by Customer Type:")
customer_type_stats = df_cleaned.groupby('customer_type').agg({
    'quantity': ['count', 'sum', 'mean'],
    'unit_price': ['mean', 'median'],
    'discount_percent': ['mean', 'median'],
    'total_price': ['sum', 'mean', 'median']
})
print(customer_type_stats)

# Cross-tabulation of product and region
print("\nCross-tabulation of Products by Region (Count):")
product_region_crosstab = pd.crosstab(df_cleaned['product'], df_cleaned['region'])
print(product_region_crosstab)

# Average purchase amount by product and customer type
print("\nAverage Purchase Amount by Product and Customer Type:")
avg_by_product_customer = df_cleaned.pivot_table(
    values='total_price',
    index='product',
    columns='customer_type',
    aggfunc='mean'
)
print(avg_by_product_customer)
print("\n" + "="*80 + "\n")

# Step 7: Identify patterns and interesting findings
print("STEP 7: Pattern Identification and Interesting Findings")

# Find the most popular product by region
most_popular_by_region = df_cleaned.groupby(['region', 'product']).size().reset_index(name='count')
most_popular_by_region = most_popular_by_region.sort_values(['region', 'count'], ascending=[True, False])
most_popular_product = most_popular_by_region.groupby('region').first()
print("\nMost Popular Product by Region:")
print(most_popular_product[['product', 'count']])

# Analyze discount patterns
print("\nDiscount Analysis by Customer Type:")
discount_analysis = df_cleaned.groupby('customer_type')['discount_percent'].value_counts().unstack().fillna(0)
print(discount_analysis)

# Find high-value purchases (top 5%)
high_value_threshold = df_cleaned['total_price'].quantile(0.95)
high_value_purchases = df_cleaned[df_cleaned['total_price'] >= high_value_threshold]
print(f"\nHigh-Value Purchases (Top 5%, Threshold: ${high_value_threshold:.2f}):")
high_value_stats = high_value_purchases.groupby('product').agg({
    'total_price': ['count', 'sum', 'mean']
})
print(high_value_stats)

# Payment method preferences by customer type
payment_preferences = pd.crosstab(
    df_cleaned['customer_type'], 
    df_cleaned['payment_method'], 
    normalize='index'
).round(2)
print("\nPayment Method Preferences by Customer Type (as proportion):")
print(payment_preferences)

# Sales trends by hour of day
df_cleaned['hour'] = df_cleaned['date'].dt.hour
hourly_sales = df_cleaned.groupby('hour')['total_price'].sum()
peak_hour = hourly_sales.idxmax()
print(f"\nPeak Sales Hour: {peak_hour}:00 with ${hourly_sales[peak_hour]:.2f} in sales")
print("\n" + "="*80 + "\n")

# Step 8: Visualize some aspects of the cleaned data and insights
print("STEP 8: Data Visualization")

# Set a consistent style for all plots
sns.set(style="whitegrid")

# Visualization 1: Sales by product
plt.figure(figsize=(10, 6))
product_sales = df_cleaned.groupby('product')['total_price'].sum().sort_values(ascending=False)
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.title('Total Sales by Product')
plt.xlabel('Product')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sales_by_product.png')

# Visualization 2: Sales by region
plt.figure(figsize=(10, 6))
region_sales = df_cleaned.groupby('region')['total_price'].sum().sort_values(ascending=False)
sns.barplot(x=region_sales.index, y=region_sales.values)
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales ($)')
plt.tight_layout()
plt.savefig('sales_by_region.png')

# Visualization 3: Average purchase by customer type
plt.figure(figsize=(10, 6))
customer_avg_purchase = df_cleaned.groupby('customer_type')['total_price'].mean().sort_values(ascending=False)
sns.barplot(x=customer_avg_purchase.index, y=customer_avg_purchase.values)
plt.title('Average Purchase Amount by Customer Type')
plt.xlabel('Customer Type')
plt.ylabel('Average Purchase ($)')
plt.tight_layout()
plt.savefig('avg_purchase_by_customer_type.png')

# Visualization 4: Distribution of purchase amounts
plt.figure(figsize=(12, 6))
sns.histplot(data=df_cleaned, x='total_price', kde=True, bins=30)
plt.title('Distribution of Purchase Amounts')
plt.xlabel('Purchase Amount ($)')
plt.ylabel('Frequency')
plt.axvline(x=df_cleaned['total_price'].mean(), color='red', linestyle='--', label=f'Mean: ${df_cleaned["total_price"].mean():.2f}')
plt.axvline(x=df_cleaned['total_price'].median(), color='green', linestyle='--', label=f'Median: ${df_cleaned["total_price"].median():.2f}')
plt.legend()
plt.tight_layout()
plt.savefig('purchase_amount_distribution.png')

# Visualization 5: Hourly sales pattern
plt.figure(figsize=(12, 6))
sns.lineplot(x=hourly_sales.index, y=hourly_sales.values)
plt.title('Sales by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Total Sales ($)')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig('hourly_sales_pattern.png')

# Visualization 6: Heatmap of product sales by region
plt.figure(figsize=(12, 8))
product_region_sales = df_cleaned.pivot_table(
    values='total_price', 
    index='product', 
    columns='region', 
    aggfunc='sum'
)
sns.heatmap(product_region_sales, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('Heatmap of Product Sales by Region')
plt.tight_layout()
plt.savefig('product_region_heatmap.png')

print("Visualizations have been created and saved as PNG files")
print("\nData analysis and insights extraction completed successfully!")
