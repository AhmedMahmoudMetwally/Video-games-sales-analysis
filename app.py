import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats
import warnings

# Suppress specific warnings from scikit-learn
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Set page config
st.set_page_config(
    page_title="Video Game Sales Analysis",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for scaler and encoder if not already present
# Ensure these are initialized as lists to avoid ValueError with st.selectbox if data isn't loaded
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler()
if 'publisher_map' not in st.session_state:
    st.session_state.publisher_map = {}
if 'columns_when_trained' not in st.session_state:
    st.session_state.columns_when_trained = []
if 'original_platforms' not in st.session_state:
    st.session_state.original_platforms = []
if 'original_genres' not in st.session_state:
    st.session_state.original_genres = []
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None


def load_data():
    """Load and return the video game sales dataset"""
    try:
        # Assuming vgsales.csv is in the same directory or accessible
        df = pd.read_csv('vgsales.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'vgsales.csv' not found. Please make sure the dataset is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}")
        return None

def clean_data(df):
    """
    Perform data cleaning operations:
    - Handle missing values
    - Remove outliers
    - Filter unrealistic years
    """
    initial_rows = df.shape[0]

    # Drop rows with missing Year
    df_cleaned_year = df.dropna(subset=['Year'])
    dropped_year_rows = initial_rows - df_cleaned_year.shape[0]
    if dropped_year_rows > 0:
        st.sidebar.info(f"Dropped {dropped_year_rows} rows with missing 'Year' values.")
    df = df_cleaned_year # Update df after dropping

    # Convert Year to integer
    df['Year'] = df['Year'].astype(int)

    # Fill Publisher missing with 'Unknown'
    # Check if there are any NaN values before filling to avoid unnecessary messages
    if df['Publisher'].isnull().any():
        st.sidebar.info("Filled missing 'Publisher' values with 'Unknown'.")
    df['Publisher'] = df['Publisher'].fillna('Unknown')


    # Remove outliers using IQR for Global_Sales
    # This step is performed because Global_Sales distribution is highly skewed with extreme outliers
    # as observed in EDA. Removing these outliers helps the linear model generalize better to the
    # majority of the data by reducing the influence of extreme values that might not represent the general trend.
    Q1 = df['Global_Sales'].quantile(0.25)
    Q3 = df['Global_Sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_clean = df[(df['Global_Sales'] >= lower_bound) & (df['Global_Sales'] <= upper_bound)]
    dropped_outlier_rows = df.shape[0] - df_clean.shape[0]
    if dropped_outlier_rows > 0:
        st.sidebar.info(f"Dropped {dropped_outlier_rows} rows identified as outliers in 'Global_Sales' using IQR method. This helps the linear model focus on typical sales ranges.")

    # Remove unrealistic years (ensure Year column is numeric)
    # This step ensures data quality by excluding records with improbable release years,
    # which could be data entry errors or placeholders.
    df_clean = df_clean[(df_clean['Year'] >= 1980) & (df_clean['Year'] <= 2020)]
    dropped_unrealistic_year_rows = (df.shape[0] - dropped_outlier_rows) - df_clean.shape[0]
    if dropped_unrealistic_year_rows > 0:
        st.sidebar.info(f"Dropped {dropped_unrealistic_year_rows} rows with 'Year' outside the 1980-2020 range (unrealistic years).")

    return df_clean

def feature_engineering(df_clean):
    """
    Perform feature engineering:
    - One-hot encoding for categorical features
    - Label encoding for Publisher (if needed, currently not used in X)
    - Feature selection
    (Note: Scaling will be done AFTER train-test split in run_analysis to prevent data leakage)
    """
    # Store unique Platform and Genre values for consistent one-hot encoding during prediction
    # Convert numpy arrays to lists for session state compatibility and robustness
    if not isinstance(df_clean['Platform'].unique(), list):
        st.session_state.original_platforms = df_clean['Platform'].unique().tolist()
    else:
        st.session_state.original_platforms = df_clean['Platform'].unique()

    if not isinstance(df_clean['Genre'].unique(), list):
        st.session_state.original_genres = df_clean['Genre'].unique().tolist()
    else:
        st.session_state.original_genres = df_clean['Genre'].unique()


    df_encoded = pd.get_dummies(df_clean, columns=['Platform', 'Genre'], drop_first=True)

    # Label encoding for Publisher - currently not used as a feature in the model's X
    # If you intend to use Publisher_encoded as a feature, add it to features_to_keep.
    st.session_state.publisher_map = {publisher: idx for idx, publisher in enumerate(df_clean['Publisher'].unique())}
    df_encoded['Publisher_encoded'] = df_clean['Publisher'].map(st.session_state.publisher_map)


    # Feature selection:
    # 'JP_Sales' is explicitly NOT included in the model's features (X) as it showed weaker correlation
    # with Global_Sales compared to 'NA_Sales' and 'EU_Sales' in EDA, and to simplify the model.
    features_to_keep = [col for col in df_encoded.columns
                        if col not in ['Name', 'Publisher', 'JP_Sales', 'Global_Sales', 'Rank', 'Publisher_encoded']] # Exclude Publisher_encoded if not explicitly used

    X = df_encoded[features_to_keep]
    y = df_encoded['Global_Sales']

    # Store columns used for training for consistent prediction
    st.session_state.columns_when_trained = X.columns.tolist()

    return X, y # Return X and y, scaling is done later

def show_model_diagnostics(y_test, y_pred):
    """
    Display regression diagnostic plots:
    - Residual plot
    - Q-Q plot
    - Residual distribution
    - Actual vs Predicted plot
    """
    residuals = y_test - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 12)) # Increased figure size

    # Residual plot
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[0, 0], alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Residual Plot (Predicted vs. Residuals)', fontsize=14)
    axes[0, 0].set_xlabel('Predicted Global Sales (Millions)', fontsize=12)
    axes[0, 0].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    axes[0, 0].grid(True, linestyle=':', alpha=0.7)

    # Q-Q plot
    stats.probplot(residuals, plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot of Residuals', fontsize=14)
    axes[0, 1].set_xlabel('Theoretical Quantiles', fontsize=12)
    axes[0, 1].set_ylabel('Ordered Residuals', fontsize=12)

    # Residual distribution
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color='skyblue', bins=50)
    axes[1, 0].set_title('Distribution of Residuals', fontsize=14)
    axes[1, 0].set_xlabel('Residuals', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].axvline(residuals.mean(), color='navy', linestyle='--', label=f'Mean: {residuals.mean():.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=':', alpha=0.7)


    # Actual vs Predicted
    sns.scatterplot(x=y_test, y=y_pred, ax=axes[1, 1], alpha=0.6, color='green')
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 1].set_title('Actual vs Predicted Global Sales', fontsize=14)
    axes[1, 1].set_xlabel('Actual Global Sales (Millions)', fontsize=12)
    axes[1, 1].set_ylabel('Predicted Global Sales (Millions)', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    st.pyplot(fig)

def run_analysis(df):
    """Run the complete analysis pipeline"""
    with st.spinner('Running analysis...'):
        # Data cleaning
        df_clean = clean_data(df.copy())

        # Feature engineering (only encoding and feature selection)
        X, y = feature_engineering(df_clean)

        # Check if X is empty after feature engineering
        if X.empty or y.empty:
            st.error("No data available for training after cleaning and feature engineering. Please check your dataset and filters.")
            return None, None, None, None, None, None, None # Removed df_encoded from return, as it's not used here

        # Split data - IMPORTANT: Split before scaling to prevent data leakage
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- Scale numerical features AFTER train-test split to prevent data leakage ---
        num_features = ['NA_Sales', 'EU_Sales', 'Other_Sales', 'Year']

        # Ensure scaler is initialized and fit only on training data
        if 'scaler' not in st.session_state:
            st.session_state.scaler = StandardScaler()

        # Check if num_features exist in X_train before fitting
        train_num_features_exist = [col for col in num_features if col in X_train.columns]
        if train_num_features_exist:
            st.session_state.scaler.fit(X_train[train_num_features_exist])
            X_train[train_num_features_exist] = st.session_state.scaler.transform(X_train[train_num_features_exist])

            # Apply transform to X_test as well
            test_num_features_exist = [col for col in num_features if col in X_test.columns]
            X_test[test_num_features_exist] = st.session_state.scaler.transform(X_test[test_num_features_exist])
        else:
            st.warning("No numerical features found to scale after cleaning. Check data or cleaning steps.")
        # -----------------------------------------------------------------------------------------

        # Train model
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Make predictions
        y_pred = lr.predict(X_test)

        # Evaluation metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Calculate Adjusted R-squared
        n = X_test.shape[0] # number of observations
        p = X_test.shape[1] # number of features
        if (n - p - 1) > 0: # Avoid division by zero or negative
            r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        else:
            r2_adj = float('nan') # Not calculable

        # Get top features
        coeff_df = pd.DataFrame({
            'Feature': X.columns, # Use X.columns from the original X before scaling (feature names remain the same)
            'Coefficient': lr.coef_
        }).sort_values('Coefficient', ascending=False)

        # Store model, data, and metrics in session state
        st.session_state.trained_model = lr
        st.session_state.r2_score = r2
        st.session_state.rmse_score = rmse
        st.session_state.r2_adj_score = r2_adj # Store R2 Adjusted
        st.session_state.coeff_df = coeff_df
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred

        # Store coefficients and intercept for regression equation
        st.session_state.model_coefficients = lr.coef_
        st.session_state.model_intercept = lr.intercept_
        st.session_state.model_features = X.columns.tolist() # Store original feature names

        return r2, rmse, r2_adj, coeff_df, y_test, y_pred, lr # Added r2_adj to return values

def predict_sales(model, input_features):
    """
    Make a prediction using the trained model
    Returns: Predicted global sales value
    """
    try:
        # Create a DataFrame for the input features
        input_df = pd.DataFrame([input_features])

        # Apply one-hot encoding for Platform and Genre based on original categories
        # Create dummy columns for all possible categories seen during training
        # This ensures consistency even if a specific platform/genre is missing in input
        for platform_cat in st.session_state.original_platforms:
            input_df[f'Platform_{platform_cat}'] = (input_df['Platform'] == platform_cat).astype(int)
        for genre_cat in st.session_state.original_genres:
            input_df[f'Genre_{genre_cat}'] = (input_df['Genre'] == genre_cat).astype(int)

        # Drop original Platform and Genre columns after one-hot encoding
        input_df = input_df.drop(columns=['Platform', 'Genre'])

        # Scale numerical features using the *fitted* scaler from session state
        # Note: JP_Sales is collected from user input but not used as a feature in the model (as per feature_engineering)
        num_features_to_scale = ['NA_Sales', 'EU_Sales', 'Other_Sales', 'Year']

        # Ensure all numerical features exist in input_df before scaling, add if missing with default 0.0
        for col in num_features_to_scale:
            if col not in input_df.columns:
                input_df[col] = 0.0 # Assign float default

        # Only transform if the scaler has been fitted
        if st.session_state.scaler is not None and hasattr(st.session_state.scaler, 'scale_'):
            input_df[num_features_to_scale] = st.session_state.scaler.transform(input_df[num_features_to_scale])
        else:
            st.error("Scaler not fitted. Please run 'Model Building' first.")
            return None


        # Ensure all columns match the training data's columns, adding missing ones with 0
        # This is critical for the model to receive input with the correct number and order of features
        final_input_df = pd.DataFrame(0, index=[0], columns=st.session_state.columns_when_trained)
        for col in st.session_state.columns_when_trained:
            if col in input_df.columns:
                final_input_df[col] = input_df[col].iloc[0] # Use .iloc[0] to get scalar value from single-row df

        # Make prediction
        prediction = model.predict(final_input_df)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}. This might happen if the model hasn't been trained yet, or if there's an issue with feature preparation. Please run 'Model Building' first.")
        return None

def main():
    """Main function to run the Streamlit app"""
    st.title("🎮 Video Game Sales Analysis & Prediction")
    st.markdown("""
    This app performs an in-depth analysis of video game sales data and utilizes a Linear Regression model
    to predict global sales based on various features like platform, genre, and regional sales.
    """)

    # Load data
    df = load_data()
    if df is None:
        return

    # Create sidebar with navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to",
                                 ["Dataset Overview",
                                  "Exploratory Analysis",
                                  "Model Building",
                                  "Make Predictions",
                                  "Results Summary"])

    if app_mode == "Dataset Overview":
        st.header("Dataset Overview")
        st.write("### Raw Dataset Information")
        st.write(f"Dataset shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
        st.write("Columns:", list(df.columns))
        st.write("First 5 rows of the dataset:")
        st.dataframe(df.head())

        st.write("### Data Types and Missing Values")
        # Display data types
        st.write("Data Types:")
        st.dataframe(df.dtypes.rename('Data Type').reset_index().rename(columns={'index': 'Column'}))

        # Display missing values
        st.write("Missing Values:")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ['Column', 'Missing Count']
        missing_df['Missing Percentage'] = (missing_df['Missing Count'] / df.shape[0]) * 100
        st.dataframe(missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False))
        if missing_df['Missing Count'].sum() == 0:
            st.info("No missing values found in the raw dataset.")
        else:
            st.info("Missing values identified in the dataset. These will be addressed in the 'Model Building' step.")


    elif app_mode == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")
        st.markdown("Dive into the data to uncover patterns, trends, and relationships. Visualizations help in understanding the distribution and interdependencies of features.")
        analysis_option = st.selectbox(
            "Select visualization:",
            ["Sales Distribution",
             "Top Selling Games",
             "Sales by Platform",
             "Sales by Genre",
             "Sales Over Time",
             "Correlation Matrix"]
        )

        if analysis_option == "Sales Distribution":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6)) # Increased figure size
            sns.histplot(df['Global_Sales'], bins=50, kde=True, ax=ax1, color='lightcoral')
            ax1.set_title('Distribution of Global Sales (Millions)', fontsize=14)
            ax1.set_xlabel('Global Sales (Millions)', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.grid(True, linestyle=':', alpha=0.7)

            sns.boxplot(x=df['Global_Sales'], ax=ax2, color='lightgreen')
            ax2.set_title('Boxplot of Global Sales (Millions)', fontsize=14)
            ax2.set_xlabel('Global Sales (Millions)', fontsize=12)
            ax2.grid(True, linestyle=':', alpha=0.7)
            st.pyplot(fig)
            st.markdown("""
            **Key Insights:**
            1. Global sales distribution is **highly right-skewed**, indicating that most games have relatively low sales.
            2. The majority of games sell less than 1 million copies globally.
            3. There are many **outliers** with exceptionally high sales (e.g., Wii Sports), which will be addressed during data cleaning to improve model performance for the typical sales range.
            """)

        elif analysis_option == "Top Selling Games":
            top_games = df[['Name', 'Platform', 'Year', 'Global_Sales']]\
                .sort_values('Global_Sales', ascending=False).head(10)
            fig = plt.figure(figsize=(12, 7)) # Increased figure size
            sns.barplot(x='Global_Sales', y='Name', data=top_games, palette='viridis')
            plt.title('Top 10 Best-Selling Games Globally', fontsize=16)
            plt.xlabel('Global Sales (Millions)', fontsize=12)
            plt.ylabel('Game Title', fontsize=12)
            plt.grid(axis='x', linestyle=':', alpha=0.7)
            st.pyplot(fig)
            st.markdown("""
            **Key Insights:**
            1. **Wii Sports** stands out significantly as the best-selling game.
            2. Many of the top-selling games are **Nintendo** exclusives or published by Nintendo, particularly for the Wii and NES platforms.
            3. The list highlights the long-term success of certain franchises like **Mario** and **Pokemon**.
            """)

        elif analysis_option == "Sales by Platform":
            platform_sales = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(15) # Top 15 platforms
            fig = plt.figure(figsize=(12, 7)) # Increased figure size
            sns.barplot(x=platform_sales.values, y=platform_sales.index, palette='crest')
            plt.title('Total Global Sales by Top 15 Platforms', fontsize=16)
            plt.xlabel('Total Global Sales (Millions)', fontsize=12)
            plt.ylabel('Platform', fontsize=12)
            plt.grid(axis='x', linestyle=':', alpha=0.7)
            st.pyplot(fig)
            st.markdown("""
            **Key Insights:**
            1. **PS2** (PlayStation 2) and **X360** (Xbox 360) have the highest cumulative global sales, followed by **PS3** (PlayStation 3) and **Wii**.
            2. Older generation consoles like **NES** and **GB** (Game Boy) still hold significant total sales due to their longevity and large install bases.
            3. This data primarily reflects sales trends up to the dataset's cutoff year (around 2016), so newer platforms like PS4/Xbox One are not fully represented.
            """)

        elif analysis_option == "Sales by Genre":
            genre_sales = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
            fig = plt.figure(figsize=(12, 7)) # Increased figure size
            sns.barplot(x=genre_sales.values, y=genre_sales.index, palette='magma')
            plt.title('Total Global Sales by Genre', fontsize=16)
            plt.xlabel('Total Global Sales (Millions)', fontsize=12)
            plt.ylabel('Genre', fontsize=12)
            plt.grid(axis='x', linestyle=':', alpha=0.7)
            st.pyplot(fig)
            st.markdown("""
            **Key Insights:**
            1. **Action** and **Sports** are the dominant genres in terms of total global sales, indicating broad appeal.
            2. **Shooter** and **Role-Playing** games also perform exceptionally well.
            3. Genres like Puzzle and Strategy have lower overall sales, suggesting a more niche market or different monetization models not captured by pure sales figures.
            """)

        elif analysis_option == "Sales Over Time":
            yearly_sales = df.groupby('Year')['Global_Sales'].sum().sort_index()
            fig = plt.figure(figsize=(12, 7)) # Increased figure size
            sns.lineplot(x=yearly_sales.index, y=yearly_sales.values, marker='o', color='darkblue')
            plt.title('Global Video Game Sales Over Time (1980-2016)', fontsize=16)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Total Global Sales (Millions)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle=':', alpha=0.7)
            st.pyplot(fig)
            st.markdown("""
            **Key Insights:**
            1. Sales grew steadily from the 1980s, experiencing a significant boom in the 2000s.
            2. Peak sales occurred around **2008-2009**, which coincides with the peak of the 7th generation console cycle (Wii, PS3, Xbox 360).
            3. A decline in sales is observed after 2009. This could be attributed to a shift towards digital distribution (not fully captured in this dataset) or changes in market dynamics. The data largely ends around 2016.
            """)

        elif analysis_option == "Correlation Matrix":
            sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
            corr_matrix = df[sales_cols].corr()
            fig = plt.figure(figsize=(9, 7)) # Increased figure size
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
            plt.title('Correlation Matrix of Regional and Global Sales', fontsize=16)
            st.pyplot(fig)
            st.markdown("""
            **Key Insights:**
            1. **NA_Sales (North America)** shows the strongest positive correlation with `Global_Sales` (0.94), indicating it's the primary driver of worldwide sales in this dataset.
            2. **EU_Sales (Europe)** also has a very strong positive correlation (0.90) with `Global_Sales`.
            3. **JP_Sales (Japan)** exhibits a weaker, but still positive, correlation (0.62) compared to NA and EU sales, suggesting the Japanese market has distinct sales patterns or a smaller overall contribution to global figures.
            4. `Other_Sales` shows a moderate correlation (0.75).
            These insights support the decision to use regional sales (excluding JP_Sales for model simplicity) as strong predictors for Global Sales.
            """)

    elif app_mode == "Model Building":
        st.header("Model Building")
        st.markdown("""
        This section builds and evaluates a Linear Regression model to predict `Global_Sales`.
        The pipeline includes data cleaning, feature engineering (one-hot encoding, feature selection),
        data splitting, scaling, model training, and evaluation.
        """)

        if st.button("Run Full Analysis"):
            r2, rmse, r2_adj, coeff_df, y_test, y_pred, lr = run_analysis(df) # Updated to receive r2_adj

            # Only proceed if analysis was successful and returned valid data
            if r2 is not None:
                st.success("Analysis completed successfully! Model trained and evaluated.")
                st.subheader("Model Performance Metrics")
                col1, col2, col3 = st.columns(3) # Added a third column for R2 Adjusted
                col1.metric("R² Score", f"{r2:.3f}")
                col2.metric("RMSE", f"{rmse:.3f}")
                if not np.isnan(r2_adj):
                    col3.metric("Adjusted R²", f"{r2_adj:.3f}")
                else:
                    col3.write("Adjusted R²: N/A (Too few observations or features for calculation)")

                st.markdown("""
                * **R² Score**: Represents the proportion of the variance in the dependent variable (Global Sales) that is predictable from the independent variables (features). A value close to 1 indicates a good fit.
                * **RMSE (Root Mean Squared Error)**: Measures the average magnitude of the errors. It tells us how concentrated the data is around the line of best fit. Lower values are better.
                * **Adjusted R²**: A modified version of R² that has been adjusted for the number of predictors in the model. It increases only if the new term improves the model more than would be expected by chance, making it a better comparison metric for models with different numbers of features.
                """)

                st.subheader("Top 10 Influential Features (Coefficients)")
                st.write("These coefficients indicate the change in `Global_Sales` (in millions) for a one-unit change in the feature, holding all other features constant.")
                st.dataframe(coeff_df.head(10))
                st.markdown("""
                * A **positive coefficient** means an increase in the feature's value leads to an increase in predicted global sales.
                * A **negative coefficient** means an increase in the feature's value leads to a decrease in predicted global sales.
                * Features related to regional sales (e.g., `NA_Sales`, `EU_Sales`, `Other_Sales`) often have the largest coefficients, indicating their strong direct relationship with `Global_Sales`.
                """)

                st.subheader("Regression Diagnostics")
                st.write("These plots help validate the assumptions of linear regression:")
                st.markdown("""
                * **Residual Plot**: Checks for linearity and homoscedasticity. Ideally, residuals should be randomly scattered around zero.
                * **Normal Q-Q Plot**: Checks if residuals are normally distributed. Points should lie close to the 45-degree line.
                * **Residual Distribution**: Visualizes the normality of residuals. Should resemble a bell curve.
                * **Actual vs Predicted Plot**: Visually assesses model fit. Points should cluster around the red dashed line (perfect prediction).
                """)
                show_model_diagnostics(st.session_state.y_test, st.session_state.y_pred)

                st.subheader("Regression Equation")
                st.write("The fitted linear regression equation is:")
                # Build the regression equation string
                equation = f"Global_Sales = {st.session_state.model_intercept:.3f}"
                for i, feature in enumerate(st.session_state.model_features):
                    coeff = st.session_state.model_coefficients[i]
                    if coeff >= 0:
                        equation += f" + {coeff:.3f} * {feature}"
                    else:
                        equation += f" - {abs(coeff):.3f} * {feature}"
                st.code(equation)
                st.markdown("""
                This equation represents the mathematical relationship learned by the model.
                Each coefficient reflects the impact of its corresponding feature on Global Sales.
                """)

                st.subheader("Multicollinearity Note")
                st.markdown("""
                Multicollinearity occurs when independent variables in a regression model are correlated.
                While not explicitly checked with VIF (Variance Inflation Factor) in this app for simplicity,
                it's an important diagnostic. High multicollinearity can make coefficient interpretations
                less reliable and inflate their standard errors, though it doesn't typically affect the model's
                overall predictive power (R²) or the fitted values. Linear Regression handles multicollinearity
                implicitly by distributing the shared variance among correlated features.
                """)

    elif app_mode == "Make Predictions":
        st.header("Make Predictions")
        st.markdown("""
        Use the trained model to predict the global sales of a new video game.
        Enter the game details below and click 'Predict Global Sales'.
        """)

        if st.session_state.trained_model is None:
            st.warning("Please run the model analysis first from the 'Model Building' tab to train the model.")
        else:
            st.subheader("Enter Game Details")

            with st.form("prediction_form"):
                col1, col2 = st.columns(2)

                # Ensure selectbox options are always lists and populated
                platform_options = st.session_state.original_platforms if st.session_state.original_platforms else df['Platform'].unique().tolist()
                genre_options = st.session_state.original_genres if st.session_state.original_genres else df['Genre'].unique().tolist()

                with col1:
                    year = st.number_input("Release Year", min_value=1980, max_value=2023, value=2010, help="The year the game was released.")
                    platform = st.selectbox("Platform", platform_options, help="The gaming console/platform.")
                    genre = st.selectbox("Genre", genre_options, help="The genre category of the game.")

                with col2:
                    na_sales = st.number_input("North America Sales (millions)", min_value=0.0, value=1.0, step=0.1, help="Sales in North America in millions of units.")
                    eu_sales = st.number_input("Europe Sales (millions)", min_value=0.0, value=0.5, step=0.1, help="Sales in Europe in millions of units.")
                    jp_sales = st.number_input("Japan Sales (millions)", min_value=0.0, value=0.3, step=0.1, help="Sales in Japan in millions of units. Note: JP_Sales is collected but not directly used in this model for prediction.")
                    other_sales = st.number_input("Other Regions Sales (millions)", min_value=0.0, value=0.2, step=0.1, help="Sales in other regions of the world in millions of units.")

                submitted = st.form_submit_button("Predict Global Sales")

            if submitted:
                # Prepare input features
                input_features = {
                    'Year': year,
                    'Platform': platform,
                    'Genre': genre,
                    'NA_Sales': na_sales,
                    'EU_Sales': eu_sales,
                    'JP_Sales': jp_sales, # Keep JP_Sales here for consistency with input form, but model doesn't use it.
                    'Other_Sales': other_sales
                }

                # Make prediction
                prediction = predict_sales(
                    st.session_state.trained_model,
                    input_features
                )

                if prediction is not None:
                    st.success(f"Predicted Global Sales: {prediction:.2f} million units")

                    # Show interpretation
                    st.subheader("Prediction Analysis")
                    st.write(f"Based on the input features, the model predicts this game will sell approximately **{prediction:.2f} million** copies worldwide.")

                    # Show impact of each feature
                    st.write("### Feature Impact on Prediction")

                    # Get model coefficients
                    if 'model_features' in st.session_state and 'model_coefficients' in st.session_state:
                        coeff_df = pd.DataFrame({
                            'Feature': st.session_state.model_features,
                            'Coefficient': st.session_state.model_coefficients
                        }).sort_values('Coefficient', key=abs, ascending=False)

                        # Show top 5 influential features
                        st.write("Top 5 features influencing this prediction:")
                        st.dataframe(coeff_df.head(5))

                        # Interpretation of top features
                        st.write("""
                        **How to interpret:**
                        - **Positive coefficients** (e.g., `NA_Sales`, `EU_Sales`) indicate that an increase in this feature's value leads to an *increase* in predicted global sales.
                        - **Negative coefficients** indicate that an increase in this feature's value leads to a *decrease* in predicted global sales (though less common for sales figures directly).
                        - **Larger absolute values** of coefficients suggest a stronger impact on the predicted sales.
                        """)
                    else:
                        st.warning("Model coefficients not available. Please run 'Model Building' first.")
                else:
                    st.error("Could not make a prediction. Please check your inputs and ensure the model is trained.")


    elif app_mode == "Results Summary":
        st.header("Results Summary")
        st.markdown("""
        This section provides a concise overview of the project, including its objectives, key findings,
        model strengths, and limitations. It's designed to summarize the entire analytical process.
        """)

        if st.session_state.trained_model is None:
            st.warning("Please run the model analysis first from the 'Model Building' tab to view the summary.")
        else:
            st.markdown("""
            ### Final Project Summary

            **Project Objective:**
            The primary objective of this project was to analyze historical video game sales data and develop a
            Linear Regression model capable of predicting global sales. Furthermore, the project aimed to identify
            key factors influencing game sales and deploy the model through a user-friendly Streamlit application.

            **Key Findings & Model Performance:**
            """)

            # Get metrics if available from session state
            if 'r2_score' in st.session_state:
                st.write(f"1. **Model Performance:** The Linear Regression model achieved an **R² score of {st.session_state.r2_score:.3f}** on the test data.")
                st.write(f"   The **RMSE (Root Mean Squared Error) is {st.session_state.rmse_score:.3f}** million units, indicating the typical error in predictions.")
                if 'r2_adj_score' in st.session_state and not np.isnan(st.session_state.r2_adj_score):
                    st.write(f"   The **Adjusted R² is {st.session_state.r2_adj_score:.3f}**, providing a more robust measure of fit considering the number of features.")
                else:
                    st.write("   Adjusted R²: Not calculable (e.g., too few observations or features).")
            else:
                st.warning("Model performance metrics not available. Please run 'Model Building' first.")


            st.markdown("""
            2. **Strongest Predictors:** Regional sales, particularly **North American Sales (`NA_Sales`)** and **European Sales (`EU_Sales`)**, were identified as the strongest predictors of global success, showing very high correlations with `Global_Sales`. `Other_Sales` also contributes significantly.
            3. **Genre Impact:** **Action** and **Sports** games consistently lead in total global sales, indicating their broad market appeal.
            4. **Platform Dominance:** Historically, platforms like **PS2**, **X360**, and **Wii** have contributed most significantly to overall sales.
            5. **Temporal Trends:** The video game industry experienced its peak sales years around **2008-2009**, followed by a decline in the subsequent years (likely due to market shifts and increased digital distribution, not fully captured by this dataset).

            **Model Strengths:**
            - **Interpretability:** Linear Regression offers high interpretability, allowing us to understand the direct impact (coefficients) of each feature on predicted sales.
            - **Predictive Power:** The R² score suggests a reasonable ability to explain the variance in global sales.
            - **Robust Preprocessing:** The pipeline includes robust data cleaning (handling missing values, outliers) and feature engineering (one-hot encoding, scaling after train-test split to prevent data leakage).
            - **Deployment Ready:** Successfully deployed via Streamlit, providing an interactive and user-friendly interface for analysis and prediction.

            **Limitations and Potential Improvements:**
            1.  **Data Currency:** The dataset mostly covers sales up to 2016. Including more recent data (post-2020) would significantly enhance the model's relevance to current market trends, especially with the rise of digital sales, mobile gaming, and new console generations.
            2.  **Model Complexity:** While Linear Regression is good for interpretability, it assumes a linear relationship. Exploring **nonlinear models** (e.g., Polynomial Regression, Ridge/Lasso Regression for regularization, or tree-based models like Random Forest/XGBoost) could potentially capture more complex patterns and improve prediction accuracy.
            3.  **Additional Features:** The model could be enhanced by incorporating external features like:
                * **Review Scores/Ratings** (e.g., Metacritic scores)
                * **Marketing Budget** and promotional spend
                * **Developer/Publisher Reputation** (beyond simple encoding)
                * **Online Play/Multiplayer features** (if quantifiable)
                * **Game Engine/Technology**
            4.  **Feature Engineering Sophistication:** More advanced feature engineering, such as creating **interaction terms** between existing features (e.g., Genre x Platform sales impact) or temporal features (e.g., game age), could yield better insights.
            5.  **Hyperparameter Tuning:** While Linear Regression has few hyperparameters, for more complex models, systematic **hyperparameter tuning** (e.g., GridSearchCV, RandomizedSearchCV) would be crucial to find optimal model configurations. For this simple Linear Regression, the default parameters generally perform well.
            6.  **Multicollinearity Handling:** While linear regression can handle multicollinearity, explicitly calculating and addressing VIF (Variance Inflation Factor) could provide deeper insights into feature independence and potentially lead to more stable coefficients if highly correlated features are causing issues.

            """)
            st.subheader("Final Regression Equation")
            st.write("The complete equation learned by the model is:")
            if 'model_intercept' in st.session_state and 'model_coefficients' in st.session_state and 'model_features' in st.session_state:
                equation = f"Global_Sales = {st.session_state.model_intercept:.3f}"
                for i, feature in enumerate(st.session_state.model_features):
                    coeff = st.session_state.model_coefficients[i]
                    if coeff >= 0:
                        equation += f" + {coeff:.3f} * {feature}"
                    else:
                        equation += f" - {abs(coeff):.3f} * {feature}"
                st.code(equation)
            else:
                st.warning("Regression equation not available. Please run 'Model Building' first.")


if __name__ == "__main__":
    main()