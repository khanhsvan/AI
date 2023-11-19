import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load merged data
merged_data = pd.read_csv('merged_data.csv')

# Rename the feature names
merged_data.rename(columns={"user_reviews": "user_reviews_new",
                             "price_final": "price_final_new",
                             "positive_ratio": "positive_ratio_new",
                             "discount": "discount_new",
                             "year_release": "year_release_new",
                             "platform": "platform_new"},
                     inplace=True)

# Use the new feature names
X = merged_data[["user_reviews_new","price_final_new","positive_ratio_new","discount_new","year_release_new","platform_new"]]
y = merged_data["positive_ratio_new"]
print(X)
# Train the model
model = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_split=10, min_samples_leaf=100, random_state=42, max_features=0.5)
model.fit(X, y)
# Save the model to a file
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(model, file)