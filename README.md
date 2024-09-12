# Cryptocurrency Price Change Prediction Using Unsupervised Learning

In this challenge, knowledge of Python and unsupervised learning was used to predict whether cryptocurrencies were affected by 24-hour or 7-day price changes. The summary statistics were obtained, and the data was plotted to visualize it before proceeding.

 <img src="https://media3.giphy.com/media/RKuFhBKHnkWjkjpnDk/giphy.webp?cid=ecf05e47nkefwg0tbsytoqp9e4d4nl6afaiay04qd9befn20&ep=v1_stickers_search&rid=giphy.webp&ct=s" class="card-img-top" alt="Project 19">

# Data Overview  
![image](https://github.com/user-attachments/assets/0955bb8d-45ee-4912-a5ca-ec93b0e3a8f5)
![image](https://github.com/user-attachments/assets/49e247be-2cc8-47af-bba8-fb3d3244e257)
![image](https://github.com/user-attachments/assets/030b4b08-3707-4b18-ad87-2140c800490d)
![image](https://github.com/user-attachments/assets/fa7b1114-71fb-46b6-ac1f-01dcb99bf510)

  
# Data Preparation
The StandardScaler() module from scikit-learn was used to normalize the data from the CSV file. A DataFrame was created with the scaled data, and the "coin_id" index from the original DataFrame was set as the index for the new DataFrame. The first five rows of the scaled DataFrame appeared as expected.  

![image](https://github.com/user-attachments/assets/23e15678-ce1d-4b73-ac78-b5b978607751)
![image](https://github.com/user-attachments/assets/c36c853f-3e19-4dda-90ff-01e930946455)

  
# Finding the Best Value for k Using the Original Scaled Data
The elbow method was used to find the best value for k by following these steps:

![image](https://github.com/user-attachments/assets/5ea7ddcd-8343-40c1-b0e9-0586f64bc195)
![image](https://github.com/user-attachments/assets/9795ce55-be13-4841-a9a8-92fb268b761e)
![image](https://github.com/user-attachments/assets/f3d79a02-eef0-4a63-955f-78647f374c55)

  
- A list with k values from 1 to 11 was created.
- An empty list was created to store the inertia values.
- A for loop was used to compute the inertia for each possible value of k.
- A dictionary with the data to plot the elbow curve was created.
  ![image](https://github.com/user-attachments/assets/7bce3022-6bb0-433b-aa38-932d3bd4ac05)
  ![image](https://github.com/user-attachments/assets/e7e72c8c-7d28-4f9c-ab71-b59c711011d0)
  ![image](https://github.com/user-attachments/assets/9a20b912-6b61-409d-b5e4-a43b791f5974)

- A line chart with the inertia values for different k values was plotted to visually identify the optimal k.
  ![image](https://github.com/user-attachments/assets/6a0bb863-27bb-4961-a405-989cf3429ea0)
  
The best value for k was determined after analyzing the elbow curve.
![image](https://github.com/user-attachments/assets/eba762ee-d341-41fd-9bc2-90acfe0afc10)

# Clustering Cryptocurrencies with K-means Using the Original Scaled Data
The steps to cluster the cryptocurrencies for the best value of k were as follows:
![image](https://github.com/user-attachments/assets/4c0c86c6-d6c9-40e9-a824-a678a96f2e79)
![image](https://github.com/user-attachments/assets/faa85740-cca1-4792-bed1-f2959031bb9a)
![image](https://github.com/user-attachments/assets/92f2d19e-c2dc-4144-af54-1d9f1405860a)
![image](https://github.com/user-attachments/assets/fca1ca91-b845-47ec-98d1-d562452883e3)
![image](https://github.com/user-attachments/assets/6699ac9b-4bdf-4308-9ed7-a34702cc2967)
![image](https://github.com/user-attachments/assets/f869c9b3-deb5-45d6-9f18-0ad5ab2c0dae)

- The K-means model was initialized with the best value for k.
- The K-means model was fitted using the original scaled DataFrame.
- The clusters were predicted to group the cryptocurrencies using the original scaled DataFrame.
- A copy of the original data was created, and a new column with the predicted clusters was added.
- A scatter plot using hvPlot was created, with the x-axis set as "price_change_percentage_24h" and the y-axis as "price_change_percentage_7d." The graph points were colored with the labels found using K-means, and the "coin_id" column was added to the hover_cols parameter to identify the cryptocurrency represented by each data point.
![image](https://github.com/user-attachments/assets/5ec20ecd-592a-4a23-856c-55cc492bf439)

# Optimizing Clusters with Principal Component Analysis (PCA)
Using the original scaled DataFrame, PCA was performed to reduce the features to three principal components. The explained variance was retrieved to determine how much information could be attributed to each principal component. The total explained variance of the three principal components was calculated. A new DataFrame with the PCA data was created, with the "coin_id" index from the original DataFrame set as the index for the new DataFrame. The first five rows of the PCA DataFrame appeared as expected.  

![image](https://github.com/user-attachments/assets/49c0f112-9307-44c1-9e3e-460e76ef0c25)
![image](https://github.com/user-attachments/assets/f6f52234-16dc-4c90-bc6b-4f84977129ac)
![image](https://github.com/user-attachments/assets/3ca8dd1d-5062-4750-8c5d-aeace10b3030)


![image](https://github.com/user-attachments/assets/15126a9f-3c43-417d-89b0-5a5950ab9743)
![image](https://github.com/user-attachments/assets/0b9e3f1f-ec8c-464e-ac3f-301b8b1886f0)
![image](https://github.com/user-attachments/assets/bfbd9924-e90e-4854-8da2-e43c192bf50b)

# Finding the Best Value for k Using the PCA Data
The elbow method was used on the PCA data to find the best value for k by following these steps:  
![image](https://github.com/user-attachments/assets/90aa1651-352e-4056-85d7-adf1effefc45)
![image](https://github.com/user-attachments/assets/5373b349-b82e-4117-971f-b184df20b98e)
![image](https://github.com/user-attachments/assets/1471ca1a-85f0-4207-aa31-287eb5ffaab7)


- A list with k values from 1 to 11 was created.
- An empty list was created to store the inertia values.
- A for loop was used to compute the inertia for each value of k.
- A dictionary with the data to plot the elbow curve was created.
- A line chart with all the inertia values for different k values was plotted to visually identify the optimal value for k.  
![image](https://github.com/user-attachments/assets/685c5dc8-f9bb-4784-9165-eb2c4aa1a5f2)

The best value for k using the PCA data was determined, and it was compared with the best k value found using the original data to assess if they differed.  
![image](https://github.com/user-attachments/assets/e4150f8c-2bce-46c6-b3ca-63eaa4e07728)
![image](https://github.com/user-attachments/assets/4dea724f-a959-4793-ba93-271e90e162a8)

# Clustering Cryptocurrencies with K-means Using the PCA Data
The following steps were taken to cluster the cryptocurrencies for the best value of k on the PCA data:  
![image](https://github.com/user-attachments/assets/1cb22c42-eb41-4c26-9f23-563fdb492d55)
![image](https://github.com/user-attachments/assets/34c7d976-19b3-41be-b834-fce23653bab9)
![image](https://github.com/user-attachments/assets/92619c82-53db-4bb7-a666-f969be337494)


- The K-means model was initialized with the best value for k.
- The K-means model was fitted using the PCA data.
- The clusters were predicted to group the cryptocurrencies using the PCA data.
- A copy of the DataFrame with the PCA data was created, and a new column was added to store the predicted clusters.
- A scatter plot using hvPlot was created, with the x-axis set as "PC1" and the y-axis as "PC2." The graph points were colored with the labels found using K-means, and the "coin_id" column was added to the hover_cols parameter to identify the cryptocurrency represented by each data point.  
![image](https://github.com/user-attachments/assets/8a26509d-99fd-4064-9606-c47a650cd2ca)

## The impact of using fewer features to cluster the data with K-Means was evaluated, and conclusions were drawn from the results.  

![image](https://github.com/user-attachments/assets/20611a4d-dec3-49b1-871c-bcd7cf8ce915)
![image](https://github.com/user-attachments/assets/e0e7c9c6-e197-490d-b690-21b1ce66dd48)
![image](https://github.com/user-attachments/assets/68118d76-1627-4d09-af80-35dc7c2cedad)

![image](https://github.com/user-attachments/assets/020b44a4-4d87-4944-8c4a-46c9ecda6110)

![image](https://github.com/user-attachments/assets/13d76b1d-d5f7-4ad0-8977-b305f5d4873d)


