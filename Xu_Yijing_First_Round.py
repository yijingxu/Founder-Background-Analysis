#!/usr/bin/env python
# coding: utf-8

# # Yijing Xu First Round Interview

# In[2]:


import pandas as pd
import numpy as np
import re


# ## Step 1: Read in Dataset

# In[3]:


df = pd.read_csv('all_member_experience.csv')


# In[4]:


print(df.shape)
df.head()


# ## Step 2: Clean Date_from and Date_to
# **Implementation:** 
# - Example inputs from date_from and date_to columns:'March 2018', 'January 2017', 'July 2015', '1997','Ekim 2011', 'Aralık 2009', 'Mayıs 2009', 'Eylül 2007', 'October 1977', 'junio de 2005'
# - The common element across all date formats in the dataset is the **year**, consistently represented in numeric form.
# - Extracting only the year simplifies the dataset, allowing for maximized data retention by avoiding the exclusion of rows due to incomplete or non-standard date information.
# - The function is applied to both `date_from` and `date_to` columns, creating new columns `year_from` and `year_to` in the DataFrame.
# 
# **Another Way:**
# - Another way of doing this is to only focus date strings in the dataset that are consistently formatted (e.g., 'March 2018', 'January 2017') and convert it to datetime object.
# Example Python Script:
# ``` python 
# match = re.search(r'(\w+) (\d{4})', data)
# if match:
#     month = match.group(1)
#     year = int(match.group(2))  # Convert year to an integer
# 
#     # Create a datetime object with the extracted month and year
#     date_obj = datetime.strptime(f'{month} {year}', '%B %Y')
# ```

# In[5]:


date_unique = df['date_from'].unique()


# In[6]:


def extract_year(date_str):
    if pd.isna(date_str) or date_str == 'nan':
        return 2023  # Default year 2023 if no valid date is provided
    year_match = re.search(r'(19|20)\d{2}', date_str) #extract four digits number from string input 
    if year_match:
        return int(year_match.group(0))  # Return as integer
    else:
        return np.nan

# Apply the function to the date_from column
df['year_from'] = df['date_from'].apply(extract_year)
df['year_to'] = df['date_to'].apply(extract_year)


# In[7]:


print(f"We have {df[df['year_from'].isna()].shape[0]} rows that contain null values after cleaning,\n"
      f"because they do not clearly state years in their answer. One example is: {df[df['year_from'].isna()]['date_from'].iloc[4]}\n"
      "I will drop these rows")


# In[8]:


df.dropna(subset=['year_from'], inplace=True)
print(df.shape)


# ## Step 3: Founder Identification
# **Assumption:** 
# Assume people who start a company would have the word 'founder' in their title.
# 
# **Implementation**
# - A Python function `is_founder` was developed to identify whether a person's title indicates they started a company.

# In[9]:


# Function to check if title indicates founding a company
# Assumption: people who start a company will have a word "founder" in their title
def is_founder(title):
    if pd.isna(title):
        return False  # Return False if the title is NaN
    return 'founder' in title.lower()

# Apply the function to create a new 'is_founder' column
df['is_founder'] = df['title'].apply(is_founder)

df_founder = df[df['is_founder'] == True].sort_values(by=['member_id', 'year_from']).drop_duplicates(subset='member_id').reset_index()


# In[10]:


df_founder = df_founder[['member_id','company_id','year_from','title', 'company_name']]
# note:len(founders_earliest['company_id'].unique())=20594,some company is founded by multiple founder in our data 

import os
file_name = 'Founder.csv'
# Check if the file already exists
if os.path.isfile(file_name):
    # Remove the old file
    os.remove(file_name)

df_founder.to_csv(file_name,index=True)


# ## Question 2: Investor Background Analysis 
# 
# To investigate investor's backgrounds. We can 
# 
# ### Step 1. Feature Engineering for Career Progression:
# 
# #### Method: 
# - **Timeline Construction:** we can extract career timelines for each founder `member_id` from `df_founder` using `year_from` and `year_to` in `df`, creating a chronological sequence of their professional experiences.
# - **Role Evolution:** we can analyze the `title` field to understand the progression of roles, espacially any shifts from technical to leadership positions, or specialization in certain areas.
# 
# #### Examples of Potiential Features: 
# 
# - **Quantitative Features:** 
#     - **Duration:** we can calculate duration (`year_to` - `year_from`) for each `member_id`'s in each title before starting a company. 
#     - **Frequency** we can calculate the frequency of changing jobs for each `member_id` before they starting a company. 
# - **Qualitative Features:**
#     - **Role before:** we can do text mining to extract key words from `title` to create a categorical column. 
#     - **Description features**: we can use LLMs to mine `description` data to extract topics, sentiments features, or embeddings from it. 
#     
#     
# #### Example Python Script:
# 
# ```python
# 
# filtered = df[df['member_id'].isin(df_founder['member_id'])]
# # to calculate each job's duration:
# filtered = filtered['year_to'] - filtered['year_from']
# # to calculate job changing frequency: 
# job_change_frequency = filtered.sort_values(by=['member_id', 'date_from']).groupby('member_id').size() - 1
# # to find role before we can do the same thing as Founder identification. we can also tokenize and stem titles
# 
# 
# ```
# 
# ### Step 2. Visualization and Statistics 
# 
# #### Visualization: 
# - We can create **Histogram** or **Boxplot** to present the distribution of durations founders spent in each role.
# - we can use **Bar Chart** or **Distribution Plot** to display the frequency of job changes for each founder.
# - We can use **Word Cloud** or **Pie Chart** to to visualize the most common roles held before becoming a founder.
# - We can utilize **Network Graph** to show founders' areas of expertise extracted from LLMs. 
# 
# #### Statistics: 
# - We can calculate mean, median, standard deviation, and range of durations in each role. This helps in understanding the typical and variation of tenure lengths before starting a company. 
# - We can do correlation analysis to the relationship between duration in previous roles and the success or scale of the company they founded (such data might available in `company_url`).
# - We can use OLS for trend analysis.
# 
# 
# ### Step 3. Analysis Methods
# 
# #### Unsupervised Learning: 
# - We can use **K-means Clustering**, which is best for quantitative features, or **Hierarchical Clustering**, which is best for quantitative and qualitative features to group inventors with similar backgrounds and identify distinctive background traits.
# - We can also use Bayesian based classification models such as Gaussian Mixture Models to incorperate prior informations.
# 
# #### Supervised Learning:
# - We can use **Tree-based Models**, such as XGboost, random forest, etc. to handle quantitative and qualitative data input effectively and make predictions on the variable we care, such as the success or scale of the company they founded.
# - We can use **Time Series Analysis**, such as ARIMA or LSTM, to analyze founder behavior over time, like career trajectory patterns leading to founding a company.
# - Regression or Classification methods can also be used if we have a clear objective to test on. 
