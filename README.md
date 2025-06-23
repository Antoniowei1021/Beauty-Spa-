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
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['funny'], label='Funny')
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


## Select Top company

| Name                        | Stars | Review Count |
|----------------------------|-------|---------------|
| Fabulous Eyebrow Threading |  5.0  |      475      |

## Select Bottom Company

| Name                   | Stars | Review Count |
|------------------------|-------|---------------|
| Golden Palace Massage  | 1.0   | 3             |

### 2 (3) What are the top keywords? 

For future sentiment analysis, it is crucial to gain insights from the review texts customers have given. But if spotting keywords directly from regular texts, most likely the results would be stopwords like "uh", "a", "the", "this", which would be meaningless for analysis. Therefore, a filter needs to be applied to the texts to remove regular stopwords to generate meaningful words. In this section, spacy is particularly helpful. 

First, remove the punctuations:
```python
import re
from collections import Counter
import csv
import string
reviews = []
with open('/Users/weichi/Desktop/beauty_spas_monthly_reviews5.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader) 
    for row in reader:
        reviews.append(row[0])

#defining the function to remove punctuation
def remove_punctuation(reviews):
    punctuationfree="".join([i for i in reviews if i not in string.punctuation])
    return punctuationfree
#storing the puntuation free text
reviews_punctuationfree=[]
for i in reviews:
    reviews_punctuationfree.append(remove_punctuation(i))
reviews_punctuationfree[:10]
lower = []
for i in reviews_punctuationfree:
    lower.append(i.lower())
lower[:1]
```
['alice jane and the entire staff here stand out above the rest of the nail salons in town  believe me ive tried a vast majority of them \nive now been going here for well over a year  i refer anyone and everyone i know here as well as bringing friends who are in town here  my mom and myself are regulars and we wouldnt have it any other way  \nalways a lovely experience a relaxed ambiance and movies  yes new and current movies as well as some classics \nabsolutely everything about stars nails is comfortable the mani stations the pedi chairs and yes all are facing the tv set for perfect movie viewing  \nclassic mani and pediatric they have them  trendy pedi treatments and facials have them as well  colors of the trendy shades abound \nrejoice  its here for you along with reasonable pricing and excellent service \ndont be ashamed how bad joss feet become boot season is here and those crest feet in wed of a wonderful pedi treatment  star nails has a pedi menu peruse it and be inclined to try all  beauty intervention here is a pleasure tuning those feet and legs  trust me feet deserve this place they do support the entire body think about it why would anyone in their right mind not treat them well \njane alice or any of the staff here will sooth moisturize with a dose of sloughing lotion and buff to perfection ahh what a relief right\nthe finale will be a slight massage for those hands or feet and then polish viola',]

Then, remove stopwords and extract the keywords:
```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
def tokenization(text):
    tokens = re.split(r'\W+', text)
    return tokens
#applying function to the column
tokenized_reviews = []
for i in lower:
    tokenized_reviews.append(tokenization(i))
tokenized_reviews[:10]

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
#defining the function to remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output
#applying function to the column
cleaned_reviews = []
for i in tokenized_reviews:
    cleaned_reviews.append(remove_stopwords(i))
cleaned_reviews[:10]
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn

wordnet_lemmatizer = WordNetLemmatizer()

#defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
#applying function to the column
lemmatized_reviews = []
for i in cleaned_reviews:
    lemmatized_reviews.append(lemmatizer(i))
lemmatized_reviews[:10]
```


[['super',
  'simple',
  'place',
  'amazing',
  'nonetheless',
  'around',
  'since',
  '30',
  'still',
  'serve',
  'thing',
  'started',
  'bologna',
  'salami',
  'sandwich',
  'mustard',
  'staff',
  'helpful',
  'friendly'],
 ['small',
  'unassuming',
  'place',
  'change',
  'menu',
  'every',
...
  'one',
  'little',
  'guy',
  'friend',
  'love']]
  
Above is just a demonstration of the process of extracting keywords. A similar process would be applied later in this project for real sentiment analysis. 

### 3 Sentimental Analysis

In this part, 2 sentiment analysis tools will be used to generate positive and negative attitudes towards different reviews. The first one is VADER and the next one is TextBlob.These two sentimental analysis tools are significantly different from each other. 

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
nltk.download('vader_lexicon')

vader = SentimentIntensityAnalyzer()
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

vader = SentimentIntensityAnalyzer()

compound_scores = []
negative_scores = []
positive_scores = []
neutral_scores = []

# Assuming `lower[:20]` is a list of preprocessed strings
for text in lower[:4000]:
    score = vader.polarity_scores(text)
    compound_scores.append(score['compound'])
    if score['compound'] >= 0.99:
        print(text)

# Plot compound score histogram
plt.figure(figsize=(8, 5))
plt.hist(compound_scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of VADER Compound Sentiment Scores')
plt.xlabel('Compound Sentiment Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

```
![image](https://github.com/user-attachments/assets/d588bc92-0bd8-4841-8f74-83d9ccd0ab55)

From the graph, it is easy to note that VADER is a very positive analysis model. It tends to skew extremely leftward and give most of the reviews a very high positive score. Though this model only examined 4000 reviews(for faster runtime), the trend was consistent throughout the whole dataset.

The next one is TextBlob:

```python
from textblob import TextBlob
import matplotlib.pyplot as plt

polarity_scores = []

for text in lower[:2000]:
    blob = TextBlob(text)
    polarity_scores.append(blob.sentiment.polarity)

# Bar plot of polarity scores (binned)
import numpy as np
bins = np.linspace(-1, 1, 21)
hist, bin_edges = np.histogram(polarity_scores, bins=bins)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

plt.figure(figsize=(10, 6))
plt.bar(bin_centers, hist, width=0.09, color='lightcoral', edgecolor='black')
plt.title('Bar Plot of TextBlob Polarity Scores')
plt.xlabel('Polarity Score (Binned)')
plt.ylabel('Frequency')
plt.grid(True, axis='y')
plt.show()
```
![image](https://github.com/user-attachments/assets/edc962bc-a416-4a40-9bee-c9dabcdb6ce3)

This time TextBlob is more normally distributed and slightly skewed to the left, with a median at 0.3. 

Next, the reviews are splitted into two groups, positive and negative. One thing to note here is that adjectives are very effective in conveying sentiments, which is spacy's specialty. Therefore, instead of using BERTopic for regular analysis, here I utilized spacy for a better result. Finally, I created a word cloud for a better view of the words. Below is the positive review part. 

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
texts = lower[:2000] 
results = classifier(texts, truncation=True, batch_size=16)
positive = []
negative = []
for text, res in zip(texts, results):
    if res['label'] == 'POSITIVE':
        positive.append(text)
    else:
        negative.append(text)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)
lower_cleaned = [clean_text(review) for review in positive]
lower_cleaned2 = [clean_text(review) for review in negative] # clean up the text for better precision

import spacy 
nlp = spacy.load('en_core_web_sm')
from wordcloud import WordCloud
from umap.umap_ import UMAP
adjectives = []
nouns = []
docs = list(nlp.pipe(lower_cleaned[:2000]))
for doc in docs:
    for token in doc:
        if token.pos_ == 'ADJ' and not token.is_stop and len(token.text) > 2:
            adjectives.append(token.lemma_.lower())
        elif token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and len(token.text) > 2:
            nouns.append(token.lemma_.lower())
umap_model = UMAP(random_state=10)  # fixed seed for reproducibility
topic_model = BERTopic(umap_model=umap_model)
topics, probs = topic_model.fit_transform(adjectives)
topic_info = topic_model.get_topic_info()

# Extract the first keyword (after topic number)
topic_info['first_word'] = topic_info['Name'].apply(lambda x: x.split('_')[1] if '_' in x else None)

# Select relevant columns
keyword_dict = dict(
    topic_info.apply(lambda row: (row['Name'].split('_')[1], row['Count']), axis=1)
)
keyword_dict = word_freq_cleaned = {v[0]: v[1] for v in keyword_dict.values()}
print(keyword_dict)
del keyword_dict['great']
del keyword_dict['good']
del keyword_dict['amazing']
del keyword_dict['nude']
wordcloud1 = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keyword_dict)

# Plot it
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Words from Reviews')
```
{'organic': 148, 'great': 700, 'good': 519, 'amazing': 402, 'friendly': 300, 'clean': 296, 'nice': 267, 'new': 224, 'happy': 202, 'sure': 135, 'professional': 125, 'beautiful': 125, 'awesome': 119, 'wonderful': 112, 'recommend': 95, 'perfect': 88, 'excellent': 86, 'able': 84, 'sweet': 78, 'long': 77, 'comfortable': 77, 'little': 74, 'regular': 72, 'busy': 71, 'worth': 66, 'right': 65, 'reasonable': 63, 'polite': 61, 'thorough': 56, 'different': 56, 'glad': 55, 'hot': 49, 'fantastic': 49, 'favorite': 48, 'bad': 47, 'melissa': 46, 'old': 45, 'fabulous': 44, 'quick': 42, 'previous': 42, 'attentive': 41, 'close': 41, 'satisfied': 41, 'soft': 39, 'cute': 39, 'hard': 38, 'extra': 38, 'late': 38, 'second': 37, 'natural': 37, 'sns': 36, 'fast': 35, 'dry': 35, 'well': 35, 'easy': 34, 'gentle': 33, 'overall': 33, 'horrible': 32, 'ill': 32, 'warm': 31, 'gorgeous': 31, 'available': 31, 'super': 31, 'pleasant': 30, 'straight': 30, 'particular': 30, 'small': 30, 'precise': 29, 'open': 29, 'technician': 29, 'exact': 29, 'free': 29, 'impressed': 28, 'short': 28, 'cheap': 28, 'acrylic': 28, 'personal': 28, 'facial': 28, 'french': 27, 'disappointed': 27, 'entire': 27, 'fair': 27, 'quiet': 27, 'big': 27, 'trendy': 27, 'affordable': 27, 'phenomenal': 26, 'patient': 26, 'real': 26, 'high': 26, 'polish': 26, 'pleased': 26, 'knowledgeable': 25, 'efficient': 25, 'nude': 24, 'talented': 24, 'modern': 24, 'thick': 23, 'helpful': 23, 'pedis': 23, 'manipedi': 23, 'south': 23, 'fancy': 23, 'superior': 23, 'excited': 23, 'single': 22, 'meticulous': 22, 'funny': 21, 'green': 21, 'kind': 21, 'detailed': 20, 'haircut': 20, 'tony': 20, 'stylist': 20, 'picky': 20, 'hesitant': 20, 'positive': 19, 'relaxed': 19, 'fresh': 19, 'live': 19, 'basic': 19, 'exceptional': 19, 'ready': 19, 'lucky': 19, 'white': 18, 'normal': 18, 'incredible': 18, 'thin': 17, 'pink': 17, 'local': 17, 'disposable': 17, 'lovely': 16, 'important': 16, 'strong': 16, 'enjoyable': 16, 'creative': 16, 'convenient': 16, 'loyal': 15, 'plenty': 15, 'interesting': 15, 'magic': 15, 'blonde': 15, 'possible': 15, 'inside': 15, 'large': 15, 'careful': 15, 'huge': 15, 'nervous': 15, 'future': 14, 'numerous': 14, 'frequent': 14, 'similar': 14, 'grateful': 14, 'awkward': 14, 'special': 14, 'spacious': 14, 'dark': 14, 'popular': 13, 'prior': 13, 'classic': 13, 'trish': 12, 'tammy': 12, 'crazy': 12, 'callus': 12, 'pricey': 12, 'caring': 12, 'brazilian': 12, 'luxurious': 12, 'essential': 11, 'desperate': 11, 'pampered': 11, 'welcome': 11, 'skeptical': 11, 'royal': 10}
Text(0.5, 1.0, 'Positive Words from Reviews')

![image](https://github.com/user-attachments/assets/70aa76fc-0015-474a-b832-b6f65e67c349)

And here is the negative word cloud:


![image](https://github.com/user-attachments/assets/be95bbd4-aa46-4191-9a80-7ef60c33225e)

This time I used nouns to point out areas that can cause negative reviews. 

Based on the two word clouds—one for positive words and one for negative words—from reviews of beauty & spa businesses, we can infer some clear strengths and weaknesses commonly mentioned by customers.

✅ Advantages (Positive Word Cloud)

1. Customer Experience & Atmosphere
Words like "friendly," "nice," "pleasant," "attentive," "professional," "comfortable," "warm," "relaxed" suggest customers appreciate:
Welcoming and respectful staff
A soothing, relaxing environment
Personalized attention
2. Quality of Service
Terms such as "clean," "perfect," "thorough," "gentle," "efficient," "precise," "professional" reflect:
High hygiene standards
Skilled and detail-oriented technicians
Reliable service delivery
3. Emotional Impact
Words like "happy," "satisfied," "awesome," "fabulous," "wonderful," "amazing" imply:
Many customers leave feeling emotionally uplifted
Businesses often exceed expectations
4. Organic/Healthy Offerings
Keywords like "organic," "natural" suggest:
Some businesses stand out for offering chemical-free or eco-conscious services

❌ Downsides (Negative Word Cloud)

1. Service Delivery & Scheduling
Prominent words include "service," "time," "appointment," "minute," "wait," "lunch," "hour," "customer":
Customers may experience long wait times or rushed appointments
Poor time management or scheduling conflicts are frequent complaints
2. Specific Services
Terms like "pedicure," "gel," "manicure," "massage," "fill," "acrylic" may point to:
Inconsistent quality across specific services
Issues with product durability (e.g., gel nails chipping early)
3. Pricing & Promotions
Words like "certificate," "coupon," "price," "discount," "groupon" indicate:
Customers might face confusion or dissatisfaction related to deal redemption
Some businesses may upsell or change terms after purchase
4. Atmosphere & Cleanliness Issues
Words like "place," "salon," "chair," "tech," "staff," "bathroom" suggest:
Mixed experiences with the physical environment or facility upkeep
Complaints about rude or inattentive technicians

### 4 Final analysis

After the sentiment analysis of the reviews, we understand what makes a good company in this industry. Now we need to find the best overall companies. 

1) Rank the top 10 companies with review counts.
```python
reviews2 = []
df = pd.read_csv('/Users/weichi/Desktop/beauty_spas_monthly_reviews6.csv')
summary = df.groupby('name').agg({
    'review_count': 'sum',
    'avg_rating': 'mean'
}).sort_values(by='review_count', ascending=False)

print(summary.head(10))
```
### Top Reviewed Beauty & Spa Businesses

| Name                   | Review Count | Average Rating |
|------------------------|--------------|----------------|
| Diamond Nails & Spa    | 622          | 4.14           |
| Polished Nails & Spa   | 508          | 3.19           |
| Elaine's Nails         | 463          | 4.40           |
| LOOK Style Society     | 376          | 4.32           |
| The Nail Bar           | 370          | 4.13           |
| Pink Nails             | 370          | 3.12           |
| Nailed and Lashed      | 352          | 3.68           |
| FINO for MEN           | 351          | 4.71           |
| 702 Nail Lounge        | 346          | 4.39           |
| Bombshell Nail & Spa   | 341          | 3.96           |

```python
df['month'] = pd.to_datetime(df['month'])

# Top 4 businesses by total review count
top_names = df.groupby('name')['review_count'].sum().nlargest(4).index.tolist()
df_top = df[df['name'].isin(top_names)]

# Monthly total review count for top businesses
monthly_total = df_top.groupby('month')['review_count'].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(monthly_total['month'], monthly_total['review_count'], marker='o')
plt.title('Total Monthly Review Count (Top 4 Nail Spas)')
plt.xlabel('Month')
plt.ylabel('Total Review Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
```

The output:

![image](https://github.com/user-attachments/assets/8e9219b3-343b-4504-bb23-49578de5e977)

```python 
# Emerging businesses (appeared after 2015)
first_seen = df.groupby('name')['month'].min().reset_index()
recent_df = df.merge(first_seen, on='name', suffixes=('', '_first'))
emerging = recent_df[recent_df['month_first'] >= '2016-01-01']

# Rank emerging businesses by avg monthly reviews
monthly_avg = emerging.groupby('name')['review_count'].mean().reset_index()
monthly_avg.columns = ['name', 'monthly_avg']
top_emerging = monthly_avg.sort_values(by='monthly_avg', ascending=False).head(10)
print(top_emerging)

# Dual-axis plot for the top business
top_business = df.groupby('name')['review_count'].sum().nlargest(1).index[0]
top_df = df[df['name'] == top_business].sort_values('month')

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_title(f'Monthly Review Count & Average Rating for {top_business}')
ax1.plot(top_df['month'], top_df['review_count'], color='blue', label='Review Count', marker='o')
ax1.set_xlabel('Month')
ax1.set_ylabel('Review Count', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
ax2.plot(top_df['month'], top_df['avg_rating'], color='orange', label='Avg Rating', marker='s')
ax2.set_ylabel('Average Rating', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
plt.xticks(rotation=45)
fig.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/30c5909d-e297-43ca-a23b-b7175c50f56a)

Finally I created a ranking system with several factors to select the best company.

```python
combined['final_score'] = (
    0.25 * combined['avg_rating_norm'] +
    0.25 * combined['growth_norm'] +
    0.20 * combined['volume_norm'] +
    0.20 * combined['sentiment_norm'] +
    0.10 * combined['loyalty_norm']
)
```
### Top 10 Ranked Businesses

| Name                | Growth  | Volume | Avg Rating | Loyalty | Final Score |
|---------------------|---------|--------|------------|---------|--------------|
| Diamond Nails & Spa | 0.078   | 622    | 4.14       | 0.91    | 0.63         |
| Elaine's Nails      | 0.054   | 463    | 4.40       | 0.95    | 0.61         |
| FINO for MEN        | 0.088   | 351    | 4.71       | 0.89    | 0.61         |
| 702 Nail Lounge     | 0.164   | 346    | 4.39       | 0.97    | 0.60         |
| LOOK Style Society  | 0.005   | 376    | 4.32       | 0.96    | 0.59         |
| Nail 21             | 2.800   | 24     | 5.00       | 0.75    | 0.58         |
| KNC Skin            | 2.000   | 10     | 5.00       | 1.00    | 0.58         |
| The Nail Bar        | 0.018   | 370    | 4.13       | 0.93    | 0.58         |
| Luxury Thai Spa     | 0.112   | 231    | 4.68       | 0.95    | 0.57         |
| Lacquer Me Up       | 0.229   | 207    | 4.78       | 0.86    | 0.56         |


As a result, there are a few advices that could be given to the employers of this industry to improve their ratings and stars:

### 1. Improve Appointment Management & Punctuality

Frequently customers complained about wait times and delays. So an online booking system is essential to keep punctuality. It can also reduce overbooking. Or Acknowledge lateness and offer partial discounts or sincere apologies.

### 2. Enhance Service Consistency Across Technicians

Some customers complained about inconsistency accross some technicians. so we need to have senior staff mentor new hires, or regularly solicit feedback from clients on individual experiences.

### 3. Boost Cleanliness and Salon Ambiance

Positive reviews frequently highlighted “clean,” “comfortable,” and “pleasant” atmospheres.

Action:

Conduct daily sanitation walkthroughs.

Invest in comfortable seating, calming décor, and soft lighting.

Add light music or offer refreshments to enhance the experience.














