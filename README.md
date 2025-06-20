# Beauty-Spa
Data analysis based on review comments on Beauty&amp;Spa data on yelp 

This project is focused on user reviews on yelp for beauty & spa business. Then apply a sentiment analysis to the reviews using spacy and nltk to extract keywords from the reviews. By generating a word cloud, analysts can find the reasons for a business to get positive responses and vice versa. 

## Introduction of the data

The dataset consists of six csv tables, each containing user data, reviews, business info, open hours, ratings, and tips. The reviews are from 2004 to late 2017.

### 1. extraction of raw data
First, connect jupyter notebook to these datasets. 
```python
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import nltk

conn = mysql.connector.connect(
    host='127.0.0.1',         
    user='root',
    password='weichi021021',
    database='yelp'  
)

```
Then extract Beauty & Spa businesses, reviews, and user data.
``` python
# Define the SQL query to get the average review stars for Beauty & Spas businesses
query = """
SELECT DISTINCT
    b.name, 
    b.stars,
    b.review_count,
    r.text,
    u.name AS user_name
FROM yelp_business b
JOIN yelp_review r ON b.business_id = r.business_id
JOIN yelp_user u ON r.user_id = u.user_id
WHERE b.categories LIKE '%Beauty & Spas%';
AND b.categories LIKE '%Beauty & Spas%' AND
    b.name IS NOT NULL AND
    r.text IS NOT NULL AND
    u.name IS NOT NULL
"""
df = pd.read_sql(query, conn)
df.to_csv('/Users/weichi/Desktop/basic_info.csv', index=False)

conn.close()
```
The result is shown below:
| name                           | stars | review_count | text                                                       | user_name |
|--------------------------------|-------|--------------|-------------------------------------------------------------|-----------|
| Great Body Massage             | 1.0   | 4            | I too have been trying to book an appt to use ...           | Julia     |
| Buff Nail Lounge               | 3.0   | 41           | Went with my friend as we had purchased the Gr...           | Julia     |
| VIKA SPA                       | 2.0   | 11           | Tried to book an appointment to use the vouche...           | Julia     |
| Splendid Nails                 | 3.0   | 9            | I started going to this nail salon last summer...           | Katherine |
| Natural Solutions Salon & Spa | 4.0   | 7            | I have since switched hairdressers and am goin...           | Katherine |
| Sephora                        | 3.0   | 20           | Seems silly to write a review about an establi...           | Katherine |
| Rain Spa & Salon               | 3.5   | 13           | Basically I have been here two times, both for...           | Katherine |
| Ray Daniels Salon              | 2.0   | 8            | Worst experience EVER! I had been there before...           | Katherine |
| Sunshine Spa & Nails          | 2.0   | 16           | Since I really needed my nails done and this p...           | Katherine |
| Essential Nails & Tanning     | 3.5   | 3            | I was in a big rush, trying to get ready for a...           | Katherine |

### 2 (1) Visualize rating distribution and review trends over time
To visualize rating, I imported matplotlib for graph generations. 
```python
import csv
import numpy as np
months = []
reviews = []

with open('/Users/weichi/Desktop/beauty_spas_monthly_reviews.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        months.append(row[0])
        reviews.append(float(row[1]))  # convert to float if you want numerical data
        months.append(row[0])
        reviews.append(float(row[1]))  # convert to float if you want numerical data
months = list(reversed(months))
reviews = list(reversed(reviews))

months_dt = [dt.datetime.strptime(m, '%Y-%m') for m in months]

plt.figure(figsize=(10, 6))
plt.plot(months_dt, reviews, color='b', label='Average Review')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=10))
months_numeric = mdates.date2num(months_dt)
x_line = np.linspace(months_numeric.min(), months_numeric.max(), 100)
y_line = 0 * x_line + 4.00

# Plot the dividing line
plt.plot(mdates.num2date(x_line), y_line, color='black', linestyle='--', linewidth=2, label='Average Line')
plt.gcf().autofmt_xdate()  
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/ed511f53-fd46-4f8f-a0e6-a7cef1f90962)

As shown in the graph, the average ratings for all the businesses is around 4.0. The tremendous shift at the left side of the graph can be interpreted as the lack of beauty & spa business in the early 2000s. Then as time goes on, the vibration stablized at 4.0. However, there was a slight dip in average review scores from 2010 to 2013. This could have several explanations:

### 1) Post-Recession Consumer Behavior
   
Lingering economic stress from the 2008 financial crisis caused consumers to be more critical of services.

People had tighter budgets, so expectations rose for service quality relative to price.

Fewer people may have been willing to tolerate mediocre service when discretionary spending was under pressure.

### 2) Business Closures and Staffing Issues

Many small Beauty & Spa businesses struggled during and after the recession:

Some cut costs, leading to fewer staff, less training, or reduced service quality.

New businesses may have opened during recovery but lacked stability or experience, contributing to lower initial reviews.

### 3) Rise in Online Review Usage
   
The period from 2010 onward saw explosive growth in Yelp and Google Reviews.

More customers started voicing negative experiences, whereas earlier online reviews may have been skewed more positively.

Increased digital transparency may have exposed service gaps

### 2 (2) Analysis around funny, useful, cool

In yelp_review table, there are 3 columns: funny, useful, and cool. The column itself is an integer, with larger value indicating more funny/useful/cool. First, extract the values of these 3 columns with a sql sentence:

```python
query = """
SELECT DISTINCT
    r.text,
    r.date,
    r.funny,
    r.useful,
    r.cool
FROM yelp_review r
JOIN yelp_business b ON r.business_id = b.business_id
WHERE b.categories LIKE '%Beauty & Spas%'
AND r.text IS NOT NULL 
ORDER BY r.date ASC
"""
df = pd.read_sql(query, conn)
df.to_csv('/Users/weichi/Desktop/funny_useful_cool.csv', index=False)

conn.close()
df = pd.read_csv('/Users/weichi/Desktop/funny_useful_cool.csv')
df['date'] = pd.to_datetime(df['date'])
months_dt = df['date'].dt.to_period('M').dt.to_timestamp()

plt.figure(figsize=(10, 6))
plt.plot(months_dt, df["funny"], color='b', label='Funny')

```
Then plot accordingly 


funny

![image](https://github.com/user-attachments/assets/5af55135-78b3-47d9-9a3f-426d35989fab)

```python
plt.plot(months_dt, df["useful"], color='g', label='Useful')
```
useful

![image](https://github.com/user-attachments/assets/a4c765f8-37b6-4141-b168-82a5664d82d5)

```python
plt.plot(months_dt, df["cool"], color='r', label='Cool')
```
cool

![image](https://github.com/user-attachments/assets/20e27897-78e6-4fd5-94f5-6f5f60f4a17b)









