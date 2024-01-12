# Founder-Background-Analysis
The python file, `Xu_Yijing_First_Round.py`, involves reading in the csv file, cleaning the date_from and date_to columns to extract years, identifying individuals who started a company based on their titles and creating a subset of data focusing on founders.
In the second section, it also proposed programming solution to understand the background of inventors before they started their company.
The Markdown documents are copied and pasted here:
## Question 1: Founder Extraction 
### Step 1: Read in Dataset
### Step 2: Clean Date_from and Date_to
**Implementation:** 
- Example inputs from date_from and date_to columns:'March 2018', 'January 2017', 'July 2015', '1997','Ekim 2011', 'Aralık 2009', 'Mayıs 2009', 'Eylül 2007', 'October 1977', 'junio de 2005'
- The common element across all date formats in the dataset is the **year**, consistently represented in numeric form.
- Extracting only the year simplifies the dataset, function: `extract_year` allowing for maximized data retention by avoiding the exclusion of rows due to incomplete or non-standard date information.
- The function is applied to both `date_from` and `date_to` columns, creating new columns `year_from` and `year_to` in the DataFrame.

**Another Way:**
- Another way of doing this is to only focus date strings in the dataset that are consistently formatted (e.g., 'March 2018', 'January 2017') and convert it to datetime object.
Example Python Script:
``` python 
match = re.search(r'(\w+) (\d{4})', data)
if match:
    month = match.group(1)
    year = int(match.group(2))  # Convert year to an integer

    # Create a datetime object with the extracted month and year
    date_obj = datetime.strptime(f'{month} {year}', '%B %Y')
```
### Step 3: Founder Identification
**Assumption:** 
Assume people who start a company would have the word 'founder' in their title.

**Implementation**
- A Python function `is_founder` was developed to identify whether a person's title indicates they started a company.



## Question 2: Investor Background Analysis 

To investigate investor's backgrounds. We can 

### Step 1. Feature Engineering for Career Progression:

#### Method: 
- **Timeline Construction:** we can extract career timelines for each founder `member_id` from `df_founder` using `year_from` and `year_to` in `df`, creating a chronological sequence of their professional experiences.
- **Role Evolution:** we can analyze the `title` field to understand the progression of roles, espacially any shifts from technical to leadership positions, or specialization in certain areas.

#### Examples of Potiential Features: 

- **Quantitative Features:** 
    - **Duration:** we can calculate duration (`year_to` - `year_from`) for each `member_id`'s in each title before starting a company. 
    - **Frequency** we can calculate the frequency of changing jobs for each `member_id` before they starting a company. 
- **Qualitative Features:**
    - **Role before:** we can do text mining to extract key words from `title` to create a categorical column. 
    - **Description features**: we can use LLMs to mine `description` data to extract topics, sentiments features, or embeddings from it. 
    
    
#### Example Python Script:

```python

filtered = df[df['member_id'].isin(df_founder['member_id'])]
# to calculate each job's duration:
filtered = filtered['year_to'] - filtered['year_from']
# to calculate job changing frequency: 
job_change_frequency = filtered.sort_values(by=['member_id', 'date_from']).groupby('member_id').size() - 1
# to find role before we can do the same thing as Founder identification. we can also tokenize and stem titles


```

### Step 2. Visualization and Statistics 

#### Visualization: 
- We can create **Histogram** or **Boxplot** to present the distribution of durations founders spent in each role.
- we can use **Bar Chart** or **Distribution Plot** to display the frequency of job changes for each founder.
- We can use **Word Cloud** or **Pie Chart** to to visualize the most common roles held before becoming a founder.
- We can utilize **Network Graph** to show founders' areas of expertise extracted from LLMs. 

#### Statistics: 
- We can calculate mean, median, standard deviation, and range of durations in each role. This helps in understanding the typical and variation of tenure lengths before starting a company. 
- We can do correlation analysis to the relationship between duration in previous roles and the success or scale of the company they founded (such data might available in `company_url`).
- We can use OLS for trend analysis.


### Step 3. Analysis Methods

#### Unsupervised Learning: 
- We can use **K-means Clustering**, which is best for quantitative features, or **Hierarchical Clustering**, which is best for quantitative and qualitative features to group inventors with similar backgrounds and identify distinctive background traits.
- We can also use Bayesian based classification models such as Gaussian Mixture Models to incorperate prior informations.

#### Supervised Learning:
- We can use **Tree-based Models**, such as XGboost, random forest, etc. to handle quantitative and qualitative data input effectively and make predictions on the variable we care, such as the success or scale of the company they founded.
- We can use **Time Series Analysis**, such as ARIMA or LSTM, to analyze founder behavior over time, like career trajectory patterns leading to founding a company.
- Regression or Classification methods can also be used if we have a clear objective to test on. 
