import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Load the CSV file into a DataFrame
file_path = r"C:\Users\valarsri\Downloads\Womens_Clothing_Statistics_Updataed1.csv"
df = pd.read_csv(file_path)

# Create new feature columns
df['Word Count'] = df['Review Text'].apply(lambda x: len(str(x).split()))
df['Sentiment Score'] = df['Review Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Select features and target variable
features = df[['Clothing ID', 'Age', 'Rating', 'Division Name', 'Class Name', 'Sentiment Score', 'Word Count', 'Department Name']]
target = df['Recommended IND']

# One-hot encode categorical features
features = pd.get_dummies(features, columns=['Division Name', 'Department Name', 'Class Name'], drop_first=True)

# Check for missing values
print("Missing values in features:")
print(features.isnull().sum())
print("\nMissing values in target:")
print(target.isnull().sum())

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

# Show data split information
print(f"\nTraining data size: {X_train.shape[0]} samples")
print(f"Testing data size: {X_test.shape[0]} samples")

# Set up logistic regression with grid search for hyperparameter tuning
model = LogisticRegression(solver='liblinear')

param_grid = {
    'C': [0.1, 1, 10],  # Reduced grid for faster processing
    'penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1')  # Reduced number of CV folds
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Predict
y_pred = best_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix with proper alignment, color range legend, and annotation
plt.figure(figsize=(8, 6))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Recommended', 'Recommended'], yticklabels=['Not Recommended', 'Recommended'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Add labels for TP, TN, FP, FN
labels = [['True Negative (TN)', 'False Positive (FP)'], ['False Negative (FN)', 'True Positive (TP)']]
for i in range(2):
    for j in range(2):
        ax.text(j + 0.5, i + 0.5, f'{labels[i][j]}\n{cm[i][j]}',
                 horizontalalignment='center',
                 verticalalignment='center',
                 color='white',
                 fontsize=12,
                 bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1))

# Adding a color bar legend
cbar = ax.collections[0].colorbar
cbar.set_label('Count')

plt.show()

# Classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Bar plot for TP, TN, FP, FN
cm_labels = ['True Negatives (TN)', 'False Positives (FP)', 'False Negatives (FN)', 'True Positives (TP)']
cm_counts = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]

# Creating a DataFrame for better handling
df_cm = pd.DataFrame({'Prediction Type': cm_labels, 'Count': cm_counts})

plt.figure(figsize=(10, 6))
barplot = sns.barplot(x='Prediction Type', y='Count', data=df_cm, palette='viridis')

# Adding data labels
for bar in barplot.patches:
    barplot.annotate(format(bar.get_height(), '.2f'),
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='center',
                     size=12, xytext=(0, 8),
                     textcoords='offset points')

plt.title('Counts of True Positives, True Negatives, False Positives, and False Negatives')
plt.ylabel('Count')
plt.xlabel('Prediction Type')
plt.xticks(rotation=45)

# Manually adding the legend handles and labels
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in sns.color_palette("viridis", n_colors=4)]
labels = cm_labels
plt.legend(handles, labels, title='Prediction Type', loc='upper left')
plt.show()

# Combine original test data with predictions
X_test_with_predictions = X_test.copy()
X_test_with_predictions['Actual'] = y_test
X_test_with_predictions['Predicted'] = y_pred

# Add back the original Division Name and Class Name for display purposes
original_data = df[['Clothing ID', 'Division Name', 'Class Name']]
X_test_with_predictions = X_test_with_predictions.merge(original_data, left_on=X_test.index, right_index=True, how='left')

# Filter for recommended products (Predicted == 1)
recommended_products = X_test_with_predictions[X_test_with_predictions['Predicted'] == 1]

# Select columns to display
recommended_products_display = recommended_products[[ 'Age', 'Rating', 'Division Name', 'Class Name']]

# Print details of recommended products
print("\nRecommended Products:")
print(recommended_products_display.to_markdown())

# Print unique class names and division names
unique_class_names = df['Class Name'].unique()
unique_division_names = df['Division Name'].unique()

# Print Division Name: Class Name list
division_class_mapping = df.groupby('Division Name')['Class Name'].unique().to_dict()
print("\nDivision Name: Class Name List")
for division, classes in division_class_mapping.items():
    print(f"{division}: {', '.join(classes)}")
