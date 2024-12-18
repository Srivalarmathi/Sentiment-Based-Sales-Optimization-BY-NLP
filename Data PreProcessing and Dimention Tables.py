import pandas as pd
import re
from textblob import TextBlob

# Load the CSV file into a DataFrame
file_path = r"Womens_Clothing_Cleaned.csv"
df = pd.read_csv(file_path)

# Define stop words
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
])

# Function to clean review text
def clean_review_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase and split into words
    words = text.lower().split()
    # Remove stop words
    words = [word for word in words if word not in stop_words]
    # Join the words back into a single string
    cleaned_text = ' '.join(words)
    return cleaned_text

# Clean the Review Text column
df['CleanedReviewText'] = df['Review Text'].apply(lambda x: clean_review_text(str(x)))

# Perform sentiment analysis on the cleaned review text
df['SentimentScore'] = df['CleanedReviewText'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Categorize sentiment as positive, neutral, or negative
def categorize_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['SentimentCategory'] = df['SentimentScore'].apply(lambda x: categorize_sentiment(x))

# Calculate Review Length and Word Count
df['ReviewLength'] = df['Review Text'].apply(len)
df['WordCount'] = df['CleanedReviewText'].apply(lambda x: len(x.split()))

# Extract unique departments, divisions, and classes
departments = df[['Department Name']].drop_duplicates().reset_index(drop=True)
departments['DepartmentID'] = departments.index + 1

divisions = df[['Division Name']].drop_duplicates().reset_index(drop=True)
divisions['DivisionID'] = divisions.index + 1

classes = df[['Class Name']].drop_duplicates().reset_index(drop=True)
classes['ClassID'] = classes.index + 1

# Merge IDs back with the original DataFrame
df = df.merge(departments, on='Department Name', how='left')
df = df.merge(divisions, on='Division Name', how='left')
df = df.merge(classes, on='Class Name', how='left')

# Create the fact table
fact_table = df[['Clothing ID', 'Age', 'Title', 'Review Text', 'Recommended IND',
                 'DepartmentID', 'Rating',
                 'DivisionID', 'ClassID', 'CleanedReviewText', 'SentimentScore',
                 'SentimentCategory', 'WordCount', 'ReviewLength']]

# Save to CSV files
fact_table.to_csv(r'fact_table.csv', index=False)
departments.to_csv(r'departments.csv', index=False)
divisions.to_csv(r'divisions.csv', index=False)
classes.to_csv(r'classes.csv', index=False)

print("Processed data saved to CSV files")
