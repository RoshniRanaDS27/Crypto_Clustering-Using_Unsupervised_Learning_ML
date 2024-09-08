# Cryptocurrency Price Change Prediction Using Unsupervised Learning
In this challenge, knowledge of Python and unsupervised learning was used to predict whether cryptocurrencies were affected by 24-hour or 7-day price changes. The summary statistics were obtained, and the data was plotted to visualize it before proceeding.
  
# Data Preparation
The StandardScaler() module from scikit-learn was used to normalize the data from the CSV file. A DataFrame was created with the scaled data, and the "coin_id" index from the original DataFrame was set as the index for the new DataFrame. The first five rows of the scaled DataFrame appeared as expected.
  
# Finding the Best Value for k Using the Original Scaled Data
The elbow method was used to find the best value for k by following these steps:

- A list with k values from 1 to 11 was created.
- An empty list was created to store the inertia values.
- A for loop was used to compute the inertia for each possible value of k.
- A dictionary with the data to plot the elbow curve was created.
- A line chart with the inertia values for different k values was plotted to visually identify the optimal k.
  
The best value for k was determined after analyzing the elbow curve.
  
# Clustering Cryptocurrencies with K-means Using the Original Scaled Data
The steps to cluster the cryptocurrencies for the best value of k were as follows:

- The K-means model was initialized with the best value for k.
- The K-means model was fitted using the original scaled DataFrame.
- The clusters were predicted to group the cryptocurrencies using the original scaled DataFrame.
- A copy of the original data was created, and a new column with the predicted clusters was added.
- A scatter plot using hvPlot was created, with the x-axis set as "price_change_percentage_24h" and the y-axis as "price_change_percentage_7d." The graph points were colored with the labels found using K-means, and the "coin_id" column was added to the hover_cols parameter to identify the cryptocurrency represented by each data point.
  
# Optimizing Clusters with Principal Component Analysis (PCA)
Using the original scaled DataFrame, PCA was performed to reduce the features to three principal components. The explained variance was retrieved to determine how much information could be attributed to each principal component. The total explained variance of the three principal components was calculated. A new DataFrame with the PCA data was created, with the "coin_id" index from the original DataFrame set as the index for the new DataFrame. The first five rows of the PCA DataFrame appeared as expected.

# Finding the Best Value for k Using the PCA Data
The elbow method was used on the PCA data to find the best value for k by following these steps:

- A list with k values from 1 to 11 was created.
- An empty list was created to store the inertia values.
- A for loop was used to compute the inertia for each value of k.
- A dictionary with the data to plot the elbow curve was created.
- A line chart with all the inertia values for different k values was plotted to visually identify the optimal value for k.
  
The best value for k using the PCA data was determined, and it was compared with the best k value found using the original data to assess if they differed.

# Clustering Cryptocurrencies with K-means Using the PCA Data
The following steps were taken to cluster the cryptocurrencies for the best value of k on the PCA data:

- The K-means model was initialized with the best value for k.
- The K-means model was fitted using the PCA data.
- The clusters were predicted to group the cryptocurrencies using the PCA data.
- A copy of the DataFrame with the PCA data was created, and a new column was added to store the predicted clusters.
- A scatter plot using hvPlot was created, with the x-axis set as "PC1" and the y-axis as "PC2." The graph points were colored with the labels found using K-means, and the "coin_id" column was added to the hover_cols parameter to identify the cryptocurrency represented by each data point.

### The impact of using fewer features to cluster the data with K-Means was evaluated, and conclusions were drawn from the results.
