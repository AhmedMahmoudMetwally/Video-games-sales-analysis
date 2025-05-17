Tasks: 
1. Dataset: 
Find a dataset containing more than 10,000 records. 
2. Exploratory Data Analysis (EDA): 
Conduct a thorough EDA to uncover patterns, anomalies, trends, and relationships within the 
data. Visualizations should be used to help understand the distribution of data and the 
relationships between features. 
3. Data Cleaning: 
This should cover issues like missing values, outliers, and inaccurate data entries. 
4. Model Development: 
Build a Linear Regression model. The model should be robust, and its parameters should be fine
tuned to get optimal performance. Evaluate the model using appropriate metrics. 
5. Regression Diagnostics: 
 Perform Regression Diagnostics to validate the model assumptions. 
 
Important Guidelines: - Make sure to split the data into training and testing datasets BEFORE anything else to avoid data 
leakage and ensure model generalization. - Avoid dropping records unless it’s extremely necessary and this should be well documented and 
justified. - You’re required to provide the complete model pipeline, from data preprocessing to final 
evaluation. 






## ✅ Overview of the Dataset

### 🔹 Loading the Data and Viewing Columns

The dataset contains **16,598 rows** and **11 columns**. Below is a short description of each column:

* **Rank**: Ranking based on global sales.
* **Name**: Name of the video game.
* **Platform**: The platform where the game was released (e.g., PS4, Xbox).
* **Year**: Release year of the game.
* **Genre**: Type of the game (e.g., Action, Sports).
* **Publisher**: The company that published the game.
* **NA\_Sales**: Sales in North America (in millions).
* **EU\_Sales**: Sales in Europe (in millions).
* **JP\_Sales**: Sales in Japan (in millions).
* **Other\_Sales**: Sales in other regions (in millions).
* **Global\_Sales**: Total worldwide sales (in millions).



### 🔹 Data Types

* **Numerical columns**: `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, `Global_Sales`, `Year`
* **Categorical columns**: `Name`, `Platform`, `Genre`, `Publisher`



### 🔹 Basic Statistics (Global Sales)

Some basic statistics for the **Global\_Sales** column:

* **Mean**: \~0.53 million units
* **Standard Deviation**: \~1.56 million units
* **Maximum**: 82.74 million units
* **Minimum**: 0.01 million units



### 🔹 Global Sales Distribution

When we plot the distribution of **Global\_Sales**, we observe that:

* Most games sold **less than 1 million units**
* A few games sold **very high numbers**
* This shows that the distribution is **right-skewed**



### 🎯 Project Goal

The main goal of this project is to **build a regression model** to **predict the global sales** of a video game based on available features such as:

* Genre
* Platform
* Year
* Publisher
