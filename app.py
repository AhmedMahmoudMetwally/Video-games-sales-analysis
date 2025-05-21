import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor # For VIF calculation
from sklearn.decomposition import PCA # Added for potential future use or discussion
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso # Added Ridge and Lasso models
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error # Added MAE
import warnings

# Suppress specific warnings from scikit-learn to keep the output clean
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Video Game Sales Analysis", # Title displayed in the browser tab
    page_icon="🎮", # Icon displayed in the browser tab
    layout="wide", # Use wide layout for better utilization of screen space
    initial_sidebar_state="expanded" # Keep the sidebar expanded by default
)

# --- Session State Initialization ---
# Streamlit's session state is used to persist variables across reruns of the script.
# This is crucial for storing the trained model, scaler, and processed data.
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler() # Scaler for numerical features
if 'publisher_map' not in st.session_state:
    st.session_state.publisher_map = {} # Map for publisher encoding (though not used in X features)
if 'columns_when_trained' not in st.session_state:
    st.session_state.columns_when_trained = [] # Stores column names after feature engineering for consistent prediction input
if 'original_platforms' not in st.session_state:
    st.session_state.original_platforms = [] # Stores unique platform names from the dataset
if 'original_genres' not in st.session_state:
    st.session_state.original_genres = [] # Stores unique genre names from the dataset
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None # Stores the trained regression model
if 'model_type' not in st.session_state:
    st.session_state.model_type = "Linear Regression" # Stores the type of model last trained
if 'data_df' not in st.session_state:
    st.session_state.data_df = None # Stores the raw loaded DataFrame
if 'eda_initialized' not in st.session_state:
    st.session_state.eda_initialized = False # Flag to ensure EDA plot initializes only once per session
if 'current_eda_option' not in st.session_state:
    st.session_state.current_eda_option = "Sales Distribution" # Default EDA plot option

# --- Data Loading Function ---
def load_data():
    """
    Loads the video game sales dataset from a CSV file.
    It attempts to read 'vgsales.csv' and stores it in Streamlit's session state.
    Handles FileNotFoundError and other exceptions gracefully.
    """
    try:
        # Assuming vgsales.csv is in the same directory as the Streamlit app script
        df = pd.read_csv('vgsales.csv')
        st.session_state.data_df = df.copy() # Store a copy of the loaded DataFrame in session state
        st.success("Dataset loaded successfully!") # Provide user feedback
        return df
    except FileNotFoundError:
        st.error("Error: 'vgsales.csv' not found. Please make sure the dataset is in the correct directory.")
        st.session_state.data_df = None # Clear data from session state if file is not found
        return None
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}") # Catch any other loading errors
        st.session_state.data_df = None # Clear data from session state on error
        return None

# --- Data Cleaning Function ---
def clean_data(df):
    """
    Performs essential data cleaning operations on the input DataFrame.
    Includes handling missing values, removing outliers, and filtering unrealistic data entries.
    """
    initial_rows = df.shape[0] # Record initial number of rows for logging

    # 1. Handle missing values in 'Year' column: Drop rows where 'Year' is missing.
    # 'Year' is critical for temporal analysis and model features.
    df_cleaned_year = df.dropna(subset=['Year'])
    dropped_year_rows = initial_rows - df_cleaned_year.shape[0]
    if dropped_year_rows > 0:
        st.sidebar.info(f"Dropped {dropped_year_rows} rows with missing 'Year' values.")
    df = df_cleaned_year # Update the DataFrame after dropping rows

    # Convert 'Year' to integer type for consistent numerical operations.
    df['Year'] = df['Year'].astype(int)

    # 2. Fill missing values in 'Publisher' column with 'Unknown'.
    # 'Publisher' is a categorical feature; 'Unknown' allows us to retain rows without losing information.
    if df['Publisher'].isnull().any(): # Check if there are any NaN values before filling
        st.sidebar.info("Filled missing 'Publisher' values with 'Unknown'.")
    df['Publisher'] = df['Publisher'].fillna('Unknown')

    # 3. Remove outliers using the Interquartile Range (IQR) method for 'Global_Sales'.
    # Global_Sales distribution is highly skewed, and extreme outliers can disproportionately
    # influence linear models. IQR method helps in focusing the model on the typical sales range.
    Q1 = df['Global_Sales'].quantile(0.25) # First quartile
    Q3 = df['Global_Sales'].quantile(0.75) # Third quartile
    IQR = Q3 - Q1 # Interquartile Range
    lower_bound = Q1 - 1.5 * IQR # Lower bound for outlier detection
    upper_bound = Q3 + 1.5 * IQR # Upper bound for outlier detection

    df_clean = df[(df['Global_Sales'] >= lower_bound) & (df['Global_Sales'] <= upper_bound)]
    dropped_outlier_rows = df.shape[0] - df_clean.shape[0]
    if dropped_outlier_rows > 0:
        st.sidebar.info(f"Dropped {dropped_outlier_rows} rows identified as outliers in 'Global_Sales' using IQR method. This helps the linear model focus on typical sales ranges.")

    # 4. Filter out unrealistic years: Keep only years within a sensible range (1980-2020).
    # This step ensures data quality by excluding potential data entry errors or future/past placeholders.
    df_clean = df_clean[(df_clean['Year'] >= 1980) & (df_clean['Year'] <= 2020)]
    # Calculate rows dropped specifically due to unrealistic years after outlier removal
    dropped_unrealistic_year_rows = (df.shape[0] - dropped_outlier_rows) - df_clean.shape[0]
    if dropped_unrealistic_year_rows > 0:
        st.sidebar.info(f"Dropped {dropped_unrealistic_year_rows} rows with 'Year' outside the 1980-2020 range (unrealistic years).")

    return df_clean

# --- Feature Engineering Function ---
def feature_engineering(df_clean):
    """
    Performs feature engineering steps including one-hot encoding for categorical variables
    and selection of features for the model.
    The scaling of numerical features is performed later, after the train-test split,
    to prevent data leakage.
    """
    # Store unique Platform and Genre values from the cleaned data.
    # These lists are used to ensure consistent one-hot encoding during prediction,
    # even if a new input game has a platform/genre not present in the training data.
    st.session_state.original_platforms = df_clean['Platform'].unique().tolist()
    st.session_state.original_genres = df_clean['Genre'].unique().tolist()

    # Apply One-Hot Encoding to 'Platform' and 'Genre' categorical features.
    # `drop_first=True` avoids multicollinearity by dropping the first category of each feature.
    df_encoded = pd.get_dummies(df_clean, columns=['Platform', 'Genre'], drop_first=True)

    # Label encoding for 'Publisher' (currently not used as a feature in the model's X).
    # This mapping is stored in session state in case it's needed for future enhancements.
    st.session_state.publisher_map = {publisher: idx for idx, publisher in enumerate(df_clean['Publisher'].unique())}
    df_encoded['Publisher_encoded'] = df_clean['Publisher'].map(st.session_state.publisher_map)

    # Feature selection: Define the features to be used in the regression model.
    # 'Name', 'Publisher', 'Rank' are irrelevant for numerical prediction.
    # 'Global_Sales' is the target variable (y).
    # 'JP_Sales' is excluded because EDA showed weaker correlation with Global_Sales compared to NA/EU sales,
    # and to simplify the model and potentially reduce multicollinearity.
    features_to_keep = [col for col in df_encoded.columns
                        if col not in ['Name', 'Publisher', 'JP_Sales', 'Global_Sales', 'Rank', 'Publisher_encoded']]

    X = df_encoded[features_to_keep] # Independent variables (features)
    y = df_encoded['Global_Sales'] # Dependent variable (target)

    # Store the exact column names used for training. This is critical for ensuring
    # that new data for prediction has the same features in the same order.
    st.session_state.columns_when_trained = X.columns.tolist()

    return X, y # Return features (X) and target (y)

# --- Multicollinearity Analysis Function (VIF) ---
def calculate_vif(X):
    """
    Calculates the Variance Inflation Factor (VIF) for each feature in a DataFrame.
    VIF measures how much the variance of an estimated regression coefficient is
    increased due to multicollinearity.
    A VIF value > 5 or > 10 typically indicates significant multicollinearity.
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    # Calculate VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values("VIF", ascending=False)

# --- Model Diagnostics Plotting Function ---
def show_model_diagnostics(y_test, y_pred):
    """
    Generates and displays a set of regression diagnostic plots using Matplotlib and Seaborn.
    These plots help in validating the assumptions of linear regression and assessing model fit.
    - Residual Plot: Checks for linearity and homoscedasticity.
    - Q-Q Plot: Checks for normality of residuals.
    - Residual Distribution: Visualizes the distribution of residuals.
    - Actual vs Predicted Plot: Visually compares actual and predicted values.
    """
    residuals = y_test - y_pred # Calculate residuals (actual - predicted)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12)) # Create a 2x2 grid of subplots

    # 1. Residual Plot (Predicted vs. Residuals)
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[0, 0], alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2) # Add a horizontal line at y=0
    axes[0, 0].set_title('Residual Plot (Predicted vs. Residuals)', fontsize=14)
    axes[0, 0].set_xlabel('Predicted Global Sales (Millions)', fontsize=12)
    axes[0, 0].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    axes[0, 0].grid(True, linestyle=':', alpha=0.7)

    # 2. Normal Q-Q Plot of Residuals
    # Compares the quantiles of the residuals to the quantiles of a theoretical normal distribution.
    # Points should lie close to the 45-degree line for normally distributed residuals.
    stats.probplot(residuals, plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot of Residuals', fontsize=14)
    axes[0, 1].set_xlabel('Theoretical Quantiles', fontsize=12)
    axes[0, 1].set_ylabel('Ordered Residuals', fontsize=12)

    # 3. Distribution of Residuals (Histogram with KDE)
    # Visualizes the frequency distribution of residuals. Ideally, it should resemble a bell curve (normal distribution).
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color='skyblue', bins=50)
    axes[1, 0].set_title('Distribution of Residuals', fontsize=14)
    axes[1, 0].set_xlabel('Residuals', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].axvline(residuals.mean(), color='navy', linestyle='--', label=f'Mean: {residuals.mean():.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=':', alpha=0.7)

    # 4. Actual vs Predicted Global Sales Plot
    # Visually assesses how well the model's predictions align with the actual values.
    # Points should cluster around the red dashed line (representing perfect prediction).
    sns.scatterplot(x=y_test, y=y_pred, ax=axes[1, 1], alpha=0.6, color='green')
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 1].set_title('Actual vs Predicted Global Sales', fontsize=14)
    axes[1, 1].set_xlabel('Actual Global Sales (Millions)', fontsize=12)
    axes[1, 1].set_ylabel('Predicted Global Sales (Millions)', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout() # Adjust subplot parameters for a tight layout
    st.pyplot(fig) # Display the Matplotlib figure in Streamlit

# --- Main Analysis and Model Training Function ---
def run_analysis(df, model_type="Linear Regression", alpha=1.0):
    """
    Runs the complete analysis pipeline: data cleaning, feature engineering,
    train-test split, scaling, model training (Linear, Ridge, or Lasso),
    cross-validation, prediction, and evaluation.
    Stores results in session state.
    """
    with st.spinner('Running analysis and training model...'): # Show a spinner while processing
        # Data preparation steps
        df_clean = clean_data(df.copy()) # Perform data cleaning
        X, y = feature_engineering(df_clean) # Perform feature engineering to get X (features) and y (target)

        # Check if data is available after preprocessing
        if X.empty or y.empty:
            st.error("No data available for training after cleaning and feature engineering. Please check your dataset and filters.")
            return None, None, None, None, None, None, None, None # Return None for all outputs

        # Split data into training and testing sets (80% train, 20% test)
        # This is crucial to evaluate the model's performance on unseen data.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale numerical features using StandardScaler.
        # Scaling is applied AFTER train-test split to prevent data leakage from the test set into the training process.
        num_features = ['NA_Sales', 'EU_Sales', 'Other_Sales', 'Year']
        # Check if numerical features exist in X_train before fitting the scaler
        train_num_features_exist = [col for col in num_features if col in X_train.columns]
        if train_num_features_exist:
            st.session_state.scaler.fit(X_train[train_num_features_exist]) # Fit scaler only on training data
            X_train[train_num_features_exist] = st.session_state.scaler.transform(X_train[train_num_features_exist]) # Transform training data
            # Apply the *fitted* scaler to the test data
            test_num_features_exist = [col for col in num_features if col in X_test.columns]
            X_test[test_num_features_exist] = st.session_state.scaler.transform(X_test[test_num_features_exist])
        else:
            st.warning("No numerical features found to scale after cleaning. Check data or cleaning steps.")

        # Model selection and training based on user's choice
        model = None
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Ridge Regression":
            model = Ridge(alpha=alpha) # Ridge adds L2 regularization
        elif model_type == "Lasso Regression":
            model = Lasso(alpha=alpha) # Lasso adds L1 regularization (can lead to feature selection)
        
        if model:
            model.fit(X_train, y_train) # Train the selected model

        # Perform Cross-Validation to assess model robustness and generalization ability.
        # Uses 5-fold cross-validation with R² as the scoring metric.
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Calculate various evaluation metrics
        r2 = r2_score(y_test, y_pred) # R-squared
        rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # Root Mean Squared Error
        mae = mean_absolute_error(y_test, y_pred) # Mean Absolute Error
        mse = mean_squared_error(y_test, y_pred) # Mean Squared Error (explicitly added)
        
        # Calculate Adjusted R-squared
        n = X_test.shape[0] # Number of observations in the test set
        p = X_test.shape[1] # Number of features in the test set
        # Formula for Adjusted R-squared: 1 - (1 - R^2) * (n - 1) / (n - p - 1)
        # Check to avoid division by zero or negative denominator
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else float('nan')
        
        # Get feature importance (coefficients) from the trained model
        # Note: For Ridge/Lasso, coefficients can be very small or zero due to regularization.
        coeff_df = pd.DataFrame({
            'Feature': X.columns, # Use original feature names from X before scaling
            'Coefficient': model.coef_ # Coefficients from the trained model
        }).sort_values('Coefficient', ascending=False)
        
        # Store all relevant results in Streamlit's session state for access across the app
        st.session_state.trained_model = model # The trained model object
        st.session_state.model_type = model_type # Type of model used
        st.session_state.r2_score = r2 # R-squared score
        st.session_state.rmse_score = rmse # RMSE score
        st.session_state.mae_score = mae # MAE score
        st.session_state.mse_score = mse # MSE score
        st.session_state.r2_adj_score = r2_adj # Adjusted R-squared score
        st.session_state.coeff_df = coeff_df # DataFrame of coefficients
        st.session_state.y_test = y_test # Actual test values
        st.session_state.y_pred = y_pred # Predicted test values
        st.session_state.model_coefficients = model.coef_ # Raw coefficients
        st.session_state.model_intercept = model.intercept_ # Raw intercept
        st.session_state.model_features = X.columns.tolist() # List of feature names
        st.session_state.cv_scores = cv_scores # Cross-validation scores
        
        return r2, rmse, mae, r2_adj, coeff_df, y_test, y_pred, model # Return key metrics and objects

# --- Prediction Function ---
def predict_sales(model, input_features):
    """
    Makes a prediction of global sales using the trained regression model.
    It prepares the input features by applying the same preprocessing steps (one-hot encoding, scaling)
    as used during model training to ensure consistency.
    Returns: Predicted global sales value, and the prepared/scaled input DataFrame.
    """
    try:
        # Create a DataFrame from the input features provided by the user
        input_df = pd.DataFrame([input_features])
        
        # Apply One-Hot Encoding for 'Platform' and 'Genre' based on the original categories
        # encountered during training. This ensures that the input DataFrame has the same
        # one-hot encoded columns as the training data, even if a specific category is missing in input.
        for platform_cat in st.session_state.original_platforms:
            input_df[f'Platform_{platform_cat}'] = (input_df['Platform'] == platform_cat).astype(int)
        for genre_cat in st.session_state.original_genres:
            input_df[f'Genre_{genre_cat}'] = (input_df['Genre'] == genre_cat).astype(int)
        
        # Drop the original 'Platform' and 'Genre' columns after one-hot encoding
        input_df = input_df.drop(columns=['Platform', 'Genre'])
        
        # Define numerical features that need to be scaled.
        # Note: 'JP_Sales' is collected from user input but is NOT used as a feature in the model
        # (as per the feature_engineering function's design).
        num_features_to_scale = ['NA_Sales', 'EU_Sales', 'Other_Sales', 'Year']
        
        # Ensure all numerical features exist in input_df before scaling.
        # If a numerical feature is missing (e.g., due to a data issue), add it with a default value of 0.0.
        for col in num_features_to_scale:
            if col not in input_df.columns:
                input_df[col] = 0.0 # Assign float default to avoid type issues
        
        # Apply the *fitted* StandardScaler (from session state) to the numerical features of the input.
        # This transforms the input data to the same scale as the training data.
        if st.session_state.scaler is not None and hasattr(st.session_state.scaler, 'scale_'):
            input_df[num_features_to_scale] = st.session_state.scaler.transform(input_df[num_features_to_scale])
        else:
            st.error("Scaler not fitted. Please run 'Model Building' first to train the model and fit the scaler.")
            return None, None # Return None for both prediction and df
        
        # Ensure the final input DataFrame for prediction has the exact same columns and order
        # as the DataFrame used during model training (`st.session_state.columns_when_trained`).
        # This is critical for the model to correctly interpret the input features.
        # Any columns present in training but not in current input_df (e.g., specific one-hot categories)
        # will be added with a value of 0.
        final_input_df = pd.DataFrame(0, index=[0], columns=st.session_state.columns_when_trained)
        for col in st.session_state.columns_when_trained:
            if col in input_df.columns:
                # Use .iloc[0] to get the scalar value from the single-row DataFrame
                final_input_df[col] = input_df[col].iloc[0]
        
        # Make the prediction using the trained model
        prediction = model.predict(final_input_df)
        return prediction[0], final_input_df # Return the single predicted value AND the prepared DataFrame
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}. This might happen if the model hasn't been trained yet, or if there's an issue with feature preparation. Please run 'Model Building' first.")
        return None, None # Return None for both prediction and df

# --- Main Streamlit Application Logic ---
def main():
    """
    Main function that orchestrates the Streamlit application.
    Manages navigation, displays different sections (Dataset Overview, EDA, Model Building, Predictions, Summary),
    and calls the relevant data processing and modeling functions.
    """
    st.title("🎮 Video Game Sales Analysis & Prediction")
    st.markdown("""
    This interactive application performs an in-depth analysis of historical video game sales data.
    It allows you to explore the dataset, build and evaluate different regression models (Linear, Ridge, Lasso)
    to predict global sales, and make new predictions based on custom inputs.
    """)

    # Load data initially or retrieve from session state
    # This ensures data is loaded once and persisted, but can be reloaded via the 'Update Data' button.
    df = st.session_state.data_df
    if df is None: # If data is not yet loaded in session state
        df = load_data() # Attempt to load it
        if df is None: # If loading fails, stop execution
            return

    # Create sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to",
                                 ["Dataset Overview",
                                  "Exploratory Analysis",
                                  "Model Building",
                                  "Make Predictions",
                                  "Results Summary"])

    # --- Section: Dataset Overview ---
    if app_mode == "Dataset Overview":
        st.header("Dataset Overview")
        st.markdown("This section provides a preliminary look at the raw dataset, including its structure, data types, and missing values.")
        
        # Add an Update button to reload the data.
        # This is useful if the underlying CSV file changes or if the user wants to re-initialize.
        if st.button("Update Data"):
            df = load_data() # Reload data when button is clicked
            if df is None: # Exit if data loading fails
                return
        
        if df is not None: # Ensure df is not None before displaying its information
            st.write("### Raw Dataset Information")
            st.write(f"Dataset shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
            st.write("Columns:", list(df.columns))
            st.write("First 5 rows of the dataset:")
            st.dataframe(df.head())

            st.write("### Data Types and Missing Values")
            # Display data types for each column
            st.write("Data Types:")
            st.dataframe(df.dtypes.rename('Data Type').reset_index().rename(columns={'index': 'Column'}))

            # Display missing values summary
            st.write("Missing Values:")
            missing_df = df.isnull().sum().reset_index()
            missing_df.columns = ['Column', 'Missing Count']
            missing_df['Missing Percentage'] = (missing_df['Missing Count'] / df.shape[0]) * 100
            # Filter to show only columns with missing values
            st.dataframe(missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False))
            if missing_df['Missing Count'].sum() == 0:
                st.info("No missing values found in the raw dataset.")
            else:
                st.info("Missing values identified in the dataset. These will be addressed during the data cleaning process in the 'Model Building' step.")
        else:
            st.warning("No dataset loaded. Please ensure 'vgsales.csv' is available in the application's directory.")

    # --- Section: Exploratory Data Analysis (EDA) ---
    elif app_mode == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")
        st.markdown("""
        Dive into the data to uncover patterns, trends, and relationships.
        Visualizations help in understanding the distribution of data and the interdependencies of features.
        """)

        if df is None:
            st.warning("Please load the dataset in 'Dataset Overview' tab first to perform EDA.")
            return

        # Automatically select "Sales Distribution" when entering EDA tab for the first time in session.
        # This ensures the first plot is shown immediately without user interaction.
        if not st.session_state.eda_initialized:
            st.session_state.eda_initialized = True
            # Set the default value for the selectbox to "Sales Distribution"
            # This ensures the selectbox reflects the auto-selected plot.
            st.session_state.current_eda_option = "Sales Distribution"
        
        # Use a key to ensure Streamlit tracks the selectbox state correctly across reruns.
        analysis_option = st.selectbox(
            "Select visualization:",
            ["Sales Distribution",
             "Top Selling Games",
             "Sales by Platform",
             "Sales by Genre",
             "Sales Over Time",
             "Correlation Matrix"],
            # Set the initial index based on the current_eda_option in session state
            index=["Sales Distribution", "Top Selling Games", "Sales by Platform", "Sales by Genre", "Sales Over Time", "Correlation Matrix"].index(st.session_state.current_eda_option),
            key="eda_visualization_selector" # Unique key for the selectbox
        )
        # Update session state if user manually changes the selection in the selectbox
        st.session_state.current_eda_option = analysis_option

        # Conditional plotting based on user's selection
        if analysis_option == "Sales Distribution":
            st.subheader("Distribution of Global Sales")
            st.markdown("This plot shows the frequency and spread of global sales, highlighting common sales ranges and outliers.")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6)) # Create a figure with two subplots
            sns.histplot(df['Global_Sales'], bins=50, kde=True, ax=ax1, color='lightcoral')
            ax1.set_title('Distribution of Global Sales (Millions)', fontsize=14)
            ax1.set_xlabel('Global Sales (Millions)', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.grid(True, linestyle=':', alpha=0.7)

            sns.boxplot(x=df['Global_Sales'], ax=ax2, color='lightgreen')
            ax2.set_title('Boxplot of Global Sales (Millions)', fontsize=14)
            ax2.set_xlabel('Global Sales (Millions)', fontsize=12)
            ax2.grid(True, linestyle=':', alpha=0.7)
            st.pyplot(fig) # Display the plot
            st.markdown("""
            **Key Insights:**
            1. Global sales distribution is **highly right-skewed**, indicating that most games have relatively low sales.
            2. The majority of games sell less than 1 million copies globally.
            3. There are many **outliers** with exceptionally high sales (e.g., Wii Sports), which will be addressed during data cleaning to improve model performance for the typical sales range.
            """)

        elif analysis_option == "Top Selling Games":
            st.subheader("Top 10 Best-Selling Games Globally")
            st.markdown("Identifies the games that have achieved the highest global sales figures.")
            top_games = df[['Name', 'Platform', 'Year', 'Global_Sales']]\
                .sort_values('Global_Sales', ascending=False).head(10)
            fig = plt.figure(figsize=(12, 7)) # Create a single figure
            sns.barplot(x='Global_Sales', y='Name', data=top_games, palette='viridis')
            plt.title('Top 10 Best-Selling Games Globally', fontsize=16)
            plt.xlabel('Global Sales (Millions)', fontsize=12)
            plt.ylabel('Game Title', fontsize=12)
            plt.grid(axis='x', linestyle=':', alpha=0.7)
            st.pyplot(fig) # Display the plot
            st.markdown("""
            **Key Insights:**
            1. **Wii Sports** stands out significantly as the best-selling game.
            2. Many of the top-selling games are **Nintendo** exclusives or published by Nintendo, particularly for the Wii and NES platforms.
            3. The list highlights the long-term success of certain franchises like **Mario** and **Pokemon**.
            """)

        elif analysis_option == "Sales by Platform":
            st.subheader("Total Global Sales by Platform")
            st.markdown("Aggregates global sales by gaming platform to show which platforms have generated the most revenue.")
            platform_sales = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(15) # Top 15 platforms
            fig = plt.figure(figsize=(12, 7)) # Create a single figure
            sns.barplot(x=platform_sales.values, y=platform_sales.index, palette='crest')
            plt.title('Total Global Sales by Top 15 Platforms', fontsize=16)
            plt.xlabel('Total Global Sales (Millions)', fontsize=12)
            plt.ylabel('Platform', fontsize=12)
            plt.grid(axis='x', linestyle=':', alpha=0.7)
            st.pyplot(fig) # Display the plot
            st.markdown("""
            **Key Insights:**
            1. **PS2** (PlayStation 2) and **X360** (Xbox 360) have the highest cumulative global sales, followed by **PS3** (PlayStation 3) and **Wii**.
            2. Older generation consoles like **NES** and **GB** (Game Boy) still hold significant total sales due to their longevity and large install bases.
            3. This data primarily reflects sales trends up to the dataset's cutoff year (around 2016), so newer platforms like PS4/Xbox One are not fully represented.
            """)

        elif analysis_option == "Sales by Genre":
            st.subheader("Total Global Sales by Genre")
            st.markdown("Examines total global sales across different game genres to identify the most popular categories.")
            genre_sales = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
            fig = plt.figure(figsize=(12, 7)) # Create a single figure
            sns.barplot(x=genre_sales.values, y=genre_sales.index, palette='magma')
            plt.title('Total Global Sales by Genre', fontsize=16)
            plt.xlabel('Total Global Sales (Millions)', fontsize=12)
            plt.ylabel('Genre', fontsize=12)
            plt.grid(axis='x', linestyle=':', alpha=0.7)
            st.pyplot(fig) # Display the plot
            st.markdown("""
            **Key Insights:**
            1. **Action** and **Sports** are the dominant genres in terms of total global sales, indicating broad appeal.
            2. **Shooter** and **Role-Playing** games also perform exceptionally well.
            3. Genres like Puzzle and Strategy have lower overall sales, suggesting a more niche market or different monetization models not captured by pure sales figures.
            """)

        elif analysis_option == "Sales Over Time":
            st.subheader("Global Video Game Sales Over Time")
            st.markdown("Tracks the trend of global video game sales annually, revealing periods of growth and decline.")
            yearly_sales = df.groupby('Year')['Global_Sales'].sum().sort_index()
            fig = plt.figure(figsize=(12, 7)) # Create a single figure
            sns.lineplot(x=yearly_sales.index, y=yearly_sales.values, marker='o', color='darkblue')
            plt.title('Global Video Game Sales Over Time (1980-2016)', fontsize=16)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Total Global Sales (Millions)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle=':', alpha=0.7)
            st.pyplot(fig) # Display the plot
            st.markdown("""
            **Key Insights:**
            1. Sales grew steadily from the 1980s, experiencing a significant boom in the 2000s.
            2. Peak sales occurred around **2008-2009**, which coincides with the peak of the 7th generation console cycle (Wii, PS3, Xbox 360).
            3. A decline in sales is observed after 2009. This could be attributed to a shift towards digital distribution (not fully captured in this dataset) or changes in market dynamics. The data largely ends around 2016.
            """)

        elif analysis_option == "Correlation Matrix":
            st.subheader("Correlation Matrix of Regional and Global Sales")
            st.markdown("Visualizes the linear relationships between regional sales figures and overall global sales.")
            sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
            corr_matrix = df[sales_cols].corr()
            fig = plt.figure(figsize=(9, 7)) # Create a single figure
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
            plt.title('Correlation Matrix of Regional and Global Sales', fontsize=16)
            st.pyplot(fig) # Display the plot
            st.markdown("""
            **Key Insights:**
            1. **NA_Sales (North America)** shows the strongest positive correlation with `Global_Sales` (0.94), indicating it's the primary driver of worldwide sales in this dataset.
            2. **EU_Sales (Europe)** also has a very strong positive correlation (0.90) with `Global_Sales`.
            3. **JP_Sales (Japan)** exhibits a weaker, but still positive, correlation (0.62) compared to NA and EU sales, suggesting the Japanese market has distinct sales patterns or a smaller overall contribution to global figures.
            4. `Other_Sales` shows a moderate correlation (0.75).
            These insights support the decision to use regional sales (excluding JP_Sales for model simplicity) as strong predictors for Global Sales.
            """)

    # --- Section: Model Building ---
    elif app_mode == "Model Building":
        st.header("Model Building")
        st.markdown("""
        This section allows you to build and evaluate different regression models to predict `Global_Sales`.
        The pipeline includes data cleaning, feature engineering, data splitting, scaling, model training,
        cross-validation, and detailed evaluation metrics.
        """)
        
        if df is None:
            st.warning("Please load the dataset in 'Dataset Overview' tab first to build the model.")
            return

        # Model selection dropdown
        model_type = st.selectbox(
            "Select Model Type:",
            ["Linear Regression", "Ridge Regression", "Lasso Regression"],
            help="Choose between standard Linear Regression or regularized versions (Ridge/Lasso) for potentially better generalization."
        )
        
        alpha = 1.0 # Default regularization strength
        if model_type != "Linear Regression":
            # Slider for regularization strength (alpha) for Ridge and Lasso models
            alpha = st.slider(
                "Regularization Strength (alpha)",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.01,
                help="Higher alpha values increase the strength of regularization, which can help prevent overfitting."
            )
        
        if st.button("Run Full Analysis"):
            # Call the run_analysis function with selected model type and alpha
            r2, rmse, mae, r2_adj, coeff_df, y_test, y_pred, model = run_analysis(df, model_type, alpha)
            
            if r2 is not None: # Proceed only if analysis was successful
                st.success("Analysis completed successfully! Model trained and evaluated.")
                
                st.subheader("Model Performance Metrics")
                st.markdown("These metrics evaluate how well the trained model predicts global sales on unseen data.")
                col1, col2, col3, col4 = st.columns(4) # Use columns for a compact display of metrics
                col1.metric("R² Score", f"{r2:.3f}") # R-squared
                col2.metric("RMSE", f"{rmse:.3f}") # Root Mean Squared Error
                col3.metric("MAE", f"{mae:.3f}") # Mean Absolute Error
                col4.metric("Adjusted R²", f"{r2_adj:.3f}" if not np.isnan(r2_adj) else "N/A") # Adjusted R-squared
                
                st.markdown("""
                * **R² Score (Coefficient of Determination)**: Represents the proportion of the variance in the dependent variable (Global Sales) that is predictable from the independent variables (features). A value closer to 1 indicates a better fit.
                * **RMSE (Root Mean Squared Error)**: Measures the average magnitude of the errors. It tells us how concentrated the data points are around the line of best fit. Lower values are better.
                * **MAE (Mean Absolute Error)**: Measures the average magnitude of the errors, similar to RMSE but less sensitive to outliers. It represents the average absolute difference between actual and predicted values. Lower values are better.
                * **Adjusted R²**: A modified version of R² that has been adjusted for the number of predictors in the model. It increases only if the new term improves the model more than would be expected by chance, making it a better comparison metric for models with different numbers of features.
                """)
                
                st.subheader("Cross-Validation Results")
                st.markdown("Cross-validation provides a more robust estimate of model performance by training and testing on different subsets of the data.")
                st.write(f"Cross-Validation R² Scores (5-fold): {st.session_state.cv_scores.round(3)}")
                st.write(f"Mean Cross-Validation R²: {np.mean(st.session_state.cv_scores):.3f}")
                st.write(f"Standard Deviation of CV R²: {np.std(st.session_state.cv_scores):.3f}")
                
                st.subheader("Multicollinearity Analysis (VIF)")
                st.markdown("""
                Multicollinearity occurs when independent variables in a regression model are highly correlated with each other.
                The Variance Inflation Factor (VIF) quantifies the severity of multicollinearity.
                * **VIF > 5** generally indicates moderate multicollinearity.
                * **VIF > 10** indicates high multicollinearity, which can make coefficient interpretations less reliable and inflate their standard errors, though it doesn't typically affect the model's overall predictive power (R²).
                """)
                # Checkbox to optionally show VIF data, as it can be large
                if st.checkbox("Show VIF for Numerical Features"):
                    numerical_features = ['NA_Sales', 'EU_Sales', 'Other_Sales', 'Year']
                    # Re-run feature engineering and train-test split to get X_train in this scope
                    df_clean = clean_data(df.copy())
                    X, y = feature_engineering(df_clean)
                    if X.empty or y.empty:
                        st.warning("No data available for VIF calculation after cleaning and feature engineering.")
                    else:
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        # Scale numerical features as in model training
                        train_num_features_exist = [col for col in numerical_features if col in X_train.columns]
                        if train_num_features_exist:
                            scaler = StandardScaler()
                            scaler.fit(X_train[train_num_features_exist])
                            X_train[train_num_features_exist] = scaler.transform(X_train[train_num_features_exist])
                        # Ensure all numerical features are present before calculating VIF
                        if all(col in X_train.columns for col in numerical_features):
                            X_numerical_scaled = X_train[numerical_features]
                            vif_data = calculate_vif(X_numerical_scaled)
                            st.dataframe(vif_data)
                        else:
                            st.warning("Could not calculate VIF: Some numerical features are missing after preprocessing.")
                
                st.subheader("Regression Diagnostics")
                st.markdown("These plots help validate the assumptions of linear regression and provide visual insights into model performance.")
                show_model_diagnostics(y_test, y_pred)
                
                st.subheader("Feature Importance (Coefficients)")
                st.markdown("These coefficients indicate the change in `Global_Sales` (in millions) for a one-unit change in the feature, holding all other features constant.")
                st.dataframe(coeff_df.head(10)) # Display top 10 coefficients
                st.markdown("""
                * A **positive coefficient** means an increase in the feature's value leads to an increase in predicted global sales.
                * A **negative coefficient** means an increase in the feature's value leads to a decrease in predicted global sales.
                * Features related to regional sales (e.g., `NA_Sales`, `EU_Sales`, `Other_Sales`) often have the largest coefficients, indicating their strong direct relationship with `Global_Sales`.
                """)
                
                st.subheader("Regression Equation")
                st.markdown("The fitted linear regression equation represents the mathematical relationship learned by the model.")
                equation = f"Global_Sales = {st.session_state.model_intercept:.3f}"
                for i, feature in enumerate(st.session_state.model_features):
                    coeff = st.session_state.model_coefficients[i]
                    if coeff >= 0:
                        equation += f" + {coeff:.3f} * {feature}"
                    else:
                        equation += f" - {abs(coeff):.3f} * {feature}"
                st.code(equation) # Display the equation in a code block

    # --- Section: Make Predictions ---
    elif app_mode == "Make Predictions":
        st.header("Make Predictions")
        st.markdown("""
        Use the trained model to predict the global sales of a new video game.
        Enter the game details below and click 'Predict Global Sales'.
        """)
        
        if st.session_state.trained_model is None:
            st.warning("Please train a model first in the 'Model Building' section to enable predictions.")
        else:
            st.write(f"Using **{st.session_state.model_type}** model for predictions.")
            
            with st.form("prediction_form"):
                st.subheader("Enter Game Details")
                col1, col2 = st.columns(2) # Use columns for a two-column layout
                
                with col1:
                    year = st.number_input("Release Year", min_value=1980, max_value=2023, value=2010, help="The year the game was released.")
                    
                    # Ensure selectbox options are always lists and populated
                    # Use original_platforms/genres from session state, which are populated after feature engineering.
                    # Fallback to df's unique values if model hasn't been trained yet (though warning prevents this path).
                    platform_options = st.session_state.original_platforms if st.session_state.original_platforms else (df['Platform'].unique().tolist() if df is not None else [])
                    genre_options = st.session_state.original_genres if st.session_state.original_genres else (df['Genre'].unique().tolist() if df is not None else [])
                    
                    platform = st.selectbox("Platform", platform_options, help="The gaming console/platform.")
                    genre = st.selectbox("Genre", genre_options, help="The genre category of the game.")
                
                with col2:
                    st.markdown("#### Regional Sales (in millions of units)")
                    na_sales = st.number_input("North America Sales (millions)", min_value=0.0, value=1.0, step=0.1, help="Sales in North America in millions of units.")
                    eu_sales = st.number_input("Europe Sales (millions)", min_value=0.0, value=0.5, step=0.1, help="Sales in Europe in millions of units.")
                    # JP_Sales is collected for user input but not used as a feature in the model (as per feature_engineering)
                    jp_sales = st.number_input("Japan Sales (millions)", min_value=0.0, value=0.3, step=0.1, help="Sales in Japan in millions of units. Note: JP_Sales is collected but not directly used in this model for prediction.")
                    other_sales = st.number_input("Other Regions Sales (millions)", min_value=0.0, value=0.2, step=0.1, help="Sales in other regions of the world in millions of units.")
                
                submitted = st.form_submit_button("Predict Global Sales")
                
                if submitted:
                    # Prepare input features dictionary
                    input_features = {
                        'Year': year,
                        'Platform': platform,
                        'Genre': genre,
                        'NA_Sales': na_sales,
                        'EU_Sales': eu_sales,
                        'JP_Sales': jp_sales, # Passed to predict_sales, but predict_sales ignores it in feature prep
                        'Other_Sales': other_sales
                    }
                    
                    # Call the prediction function, now it returns both prediction and the prepared DF
                    predicted_sales, final_input_df_for_impact = predict_sales(st.session_state.trained_model, input_features)
                    if predicted_sales is not None:
                        st.success(f"Predicted Global Sales: **{predicted_sales:.2f} million units**.")
                        
                        # Show how each feature contributed to the prediction
                        # Ensure final_input_df_for_impact is not None before proceeding
                        if 'model_features' in st.session_state and 'model_coefficients' in st.session_state and final_input_df_for_impact is not None:
                            st.write("### Feature Impacts on Prediction")
                            st.markdown("This table shows the contribution of each input feature to the final predicted sales value.")
                            impacts = {}
                            
                            # Calculate impacts for each feature using the already prepared and scaled final_input_df_for_impact
                            for i, feature_name in enumerate(st.session_state.model_features):
                                coeff = st.session_state.model_coefficients[i]
                                # Get the scaled value of the feature from the prepared DataFrame
                                feature_value_scaled = final_input_df_for_impact[feature_name].iloc[0]
                                impacts[feature_name] = coeff * feature_value_scaled
                            
                            # Add intercept's contribution
                            if 'model_intercept' in st.session_state:
                                impacts['Intercept'] = st.session_state.model_intercept

                            impacts_df = pd.DataFrame.from_dict(impacts, orient='index', columns=['Impact'])
                            st.dataframe(impacts_df.sort_values('Impact', ascending=False))


    # --- Section: Results Summary ---
    elif app_mode == "Results Summary":
        st.header("Project Results Summary")
        st.markdown("""
        This section provides a concise summary of the project's key findings, model performance, and insights.
        It encapsulates the overall success and limitations of the analysis and predictive modeling.
        """)
        
        if st.session_state.trained_model is None:
            st.warning("No model results available. Please run the analysis first in the 'Model Building' section.")
        else:
            st.markdown("## Final Project Summary\n")
            st.markdown("**Project Goal:**\n")
            st.markdown("""
            The primary objective of this project is to analyze historical video game sales data and develop a Linear Regression model capable of predicting global sales. Furthermore, the project aims to identify key factors influencing game sales and deploy the model through a user-friendly Streamlit application.
            """)

            st.markdown("\n**Key Findings & Model Performance:**\n")
            if 'r2_score' in st.session_state and not np.isnan(st.session_state.r2_score):
                st.markdown(f"**Model Type Used:** `{st.session_state.model_type}`")
                st.markdown(f"1. **Model Performance:** The trained `{st.session_state.model_type}` model achieved the following metrics on the test data:")
                st.markdown(f"   - **R² Score (Coefficient of Determination):** `{st.session_state.r2_score:.3f}`")
                st.markdown(f"   - **Adjusted R² (Adjusted Coefficient of Determination):** `{st.session_state.r2_adj_score:.3f}`" if not np.isnan(st.session_state.r2_adj_score) else "   - **Adjusted R²:** `N/A` (Too few observations or features for calculation)")
                st.markdown(f"   - **RMSE (Root Mean Squared Error):** `{st.session_state.rmse_score:.3f}` million units. This indicates the typical magnitude of prediction errors.")
                st.markdown(f"   - **MAE (Mean Absolute Error):** `{st.session_state.mae_score:.3f}` million units. This represents the average absolute difference between actual and predicted sales.")
                st.markdown(f"   - **MSE (Mean Squared Error):** `{st.session_state.mse_score:.3f}` million units. This is the average of the squared differences between actual and predicted values.")
                
                st.markdown(f"\n   **Cross-Validation (5-fold) Mean R²:** `{np.mean(st.session_state.cv_scores):.3f}` (Standard Deviation: `{np.std(st.session_state.cv_scores):.3f}`). Cross-validation provides a more reliable estimate of the model's generalization performance.")

            else:
                st.markdown("Model performance metrics are not available. Please run 'Model Building' first to populate them.")


            st.markdown("""
            2. **Strongest Predictors:** Regional sales, particularly **North America Sales (`NA_Sales`)** and **Europe Sales (`EU_Sales`)**, were identified as the strongest predictors of global success, showing very high correlations with `Global_Sales`. `Other_Sales` also contributed significantly.
            3. **Genre Impact:** **Action** and **Sports** genres consistently lead in total global sales, indicating their widespread market appeal.
            4. **Platform Dominance:** Historically, platforms like **PS2**, **X360**, and **Wii** have contributed significantly to overall sales.
            5. **Temporal Trends:** The video game industry saw a peak in sales around **2008-2009**, followed by a decline in subsequent years (likely due to market shifts and increased digital distribution, which is not fully captured by this dataset).

            **Model Strengths:**
            - **Interpretability:** Linear Regression (and its regularized variants) offers high interpretability, allowing us to understand the direct impact (coefficients) of each feature on predicted sales.
            - **Predictive Power:** The R² score indicates a reasonable ability to explain the variance in global sales.
            - **Robust Preprocessing:** The pipeline includes robust data cleaning (missing values, outliers) and feature engineering (one-hot encoding, scaling after train-test split to prevent data leakage).
            - **Deployment Ready:** Successfully deployed via Streamlit, providing an interactive and user-friendly interface for analysis and prediction.

            **Limitations and Potential Improvements:**
            1.  **Data Recency:** The dataset covers most sales up to 2016. Including more recent data (post-2020) would significantly enhance the model's relevance to current market trends, especially with the rise of digital sales, mobile gaming, and newer console generations.
            2.  **Model Complexity:** While Linear Regression is good for interpretability, it assumes a linear relationship. Exploring **non-linear models** (e.g., Polynomial Regression, tree-based models like Random Forest/XGBoost, or neural networks) could capture more complex patterns and potentially improve prediction accuracy.
            3.  **Additional Features:** The model could be enhanced by incorporating external features such as:
                * **Review scores/ratings** (e.g., Metacritic scores)
                * **Marketing budget** and promotional spending
                * **Developer/Publisher reputation** (beyond simple encoding)
                * **Online/Multiplayer features** (if quantifiable)
                * **Game engine/technology**
            4.  **Advanced Feature Engineering:** More advanced feature engineering, such as creating **interaction terms** between existing features (e.g., Genre × Platform sales effect) or temporal features (e.g., game's age), could lead to better insights.
            5.  **Hyperparameter Tuning:** For regularized models (Ridge, Lasso) or more complex models, systematic **hyperparameter tuning** (e.g., GridSearchCV, RandomizedSearchCV) would be crucial to find optimal model configurations.
            6.  **Multicollinearity Handling:** While VIF is shown, a more rigorous strategy for dealing with high multicollinearity (e.g., principal component regression if PCA is used, or explicit feature removal/combination) could be explored if it significantly impacts coefficient stability or interpretability.
            """)

            st.markdown("\n### Final Regression Equation\n")
            if 'model_intercept' in st.session_state and st.session_state.model_intercept is not None:
                equation = f"Global_Sales = {st.session_state.model_intercept:.3f}"
                for i, feature in enumerate(st.session_state.model_features):
                    coeff = st.session_state.model_coefficients[i]
                    if coeff >= 0:
                        equation += f" + {coeff:.3f} * {feature}"
                    else:
                        equation += f" - {abs(coeff):.3f} * {feature}"
                st.code(equation)
            else:
                st.markdown("Regression equation is not available. Please run 'Model Building' first.")


# --- Entry point for the Streamlit application ---
if __name__ == "__main__":
    main()
