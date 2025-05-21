import gradio as gr
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
import io

# Suppress specific warnings from scikit-learn
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Global variables to store trained model, scaler, and data for consistency across Gradio functions.
# Gradio functions are stateless by default, so global variables are used to maintain application state.
global_df = None
trained_model = None
scaler = StandardScaler()
publisher_map = {}
columns_when_trained = []
original_platforms = [] # To store unique platforms from the loaded data
original_genres = []    # To store unique genres from the loaded data
model_coefficients = None
model_intercept = None
model_features = []
# Store metrics globally for summary tab, as it's hard to pass them directly
global_r2_score = np.nan
global_rmse_score = np.nan
global_r2_adj_score = np.nan


# --- Data Loading and Preprocessing Functions ---

def load_data_gradio(file):
    """
    Loads the video game sales dataset from a Gradio File object.
    Updates the global_df and returns formatted dataset information.
    """
    global global_df
    if file is None:
        return pd.DataFrame(), "Please upload a CSV file."
    try:
        # file.name contains the path to the temporary file Gradio created
        df = pd.read_csv(file.name)
        global_df = df.copy() # Store for later use in other functions

        # Prepare initial info for display
        initial_info = f"Dataset shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})\n"
        initial_info += "Columns: " + ", ".join(list(df.columns)) + "\n"
        initial_info += "\n### First 5 rows of the dataset:\n"
        initial_info += df.head().to_markdown(index=False) # Use to_markdown for better display in Gradio Markdown component
        initial_info += "\n\n### Data Types and Missing Values\n"
        initial_info += "Data Types:\n"
        initial_info += df.dtypes.rename('Data Type').reset_index().rename(columns={'index': 'Column'}).to_markdown(index=False)

        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ['Column', 'Missing Count']
        missing_df['Missing Percentage'] = (missing_df['Missing Count'] / df.shape[0]) * 100
        missing_values_info = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False).to_markdown(index=False)

        if missing_df['Missing Count'].sum() == 0:
            missing_values_text = "\n\nNo missing values found in the raw dataset."
        else:
            missing_values_text = "\n\nMissing values identified in the dataset. These will be addressed in the 'Model Building' step:\n" + missing_values_info

        # Return the dataframe (as a state) and the formatted info string
        return df, initial_info + missing_values_text
    except FileNotFoundError:
        return pd.DataFrame(), "Error: CSV file not found. Please ensure the dataset is uploaded."
    except Exception as e:
        return pd.DataFrame(), f"Failed to load dataset: {str(e)}"

def clean_data(df):
    """
    Performs data cleaning operations:
    - Handles missing values
    - Removes outliers
    - Filters unrealistic years
    Returns cleaned DataFrame and a log of cleaning actions.
    """
    initial_rows = df.shape[0]
    cleaning_log = []

    # Drop rows with missing 'Year'
    df_cleaned_year = df.dropna(subset=['Year'])
    dropped_year_rows = initial_rows - df_cleaned_year.shape[0]
    if dropped_year_rows > 0:
        cleaning_log.append(f"Dropped {dropped_year_rows} rows with missing 'Year' values.")
    df = df_cleaned_year # Update df after dropping

    # Convert Year to integer
    df['Year'] = df['Year'].astype(int)

    # Fill Publisher missing with 'Unknown'
    if df['Publisher'].isnull().any():
        cleaning_log.append("Filled missing 'Publisher' values with 'Unknown'.")
    df['Publisher'] = df['Publisher'].fillna('Unknown')

    # Remove outliers using IQR for Global_Sales
    Q1 = df['Global_Sales'].quantile(0.25)
    Q3 = df['Global_Sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_clean = df[(df['Global_Sales'] >= lower_bound) & (df['Global_Sales'] <= upper_bound)]
    dropped_outlier_rows = df.shape[0] - df_clean.shape[0]
    if dropped_outlier_rows > 0:
        cleaning_log.append(f"Dropped {dropped_outlier_rows} rows identified as outliers in 'Global_Sales' using IQR method. This helps the linear model focus on typical sales ranges.")

    # Remove unrealistic years (ensure Year column is numeric)
    df_clean = df_clean[(df_clean['Year'] >= 1980) & (df_clean['Year'] <= 2020)]
    dropped_unrealistic_year_rows = (df.shape[0] - dropped_outlier_rows) - df_clean.shape[0]
    if dropped_unrealistic_year_rows > 0:
        cleaning_log.append(f"Dropped {dropped_unrealistic_year_rows} rows with 'Year' outside the 1980-2020 range (unrealistic years).")

    return df_clean, "\n".join(cleaning_log)

def feature_engineering(df_clean):
    """
    Performs feature engineering:
    - One-hot encoding for categorical features
    - Label encoding for Publisher (not used in X, but available)
    - Feature selection
    Updates global variables for consistent prediction.
    """
    global original_platforms, original_genres, publisher_map, columns_when_trained

    # Get unique platforms and genres from the cleaned data to use in prediction dropdowns
    original_platforms = df_clean['Platform'].unique().tolist()
    original_genres = df_clean['Genre'].unique().tolist()

    df_encoded = pd.get_dummies(df_clean, columns=['Platform', 'Genre'], drop_first=True)

    publisher_map = {publisher: idx for idx, publisher in enumerate(df_clean['Publisher'].unique())}
    df_encoded['Publisher_encoded'] = df_clean['Publisher'].map(publisher_map)

    # Feature selection: 'JP_Sales' is explicitly NOT included in the model's features (X)
    features_to_keep = [col for col in df_encoded.columns
                        if col not in ['Name', 'Publisher', 'JP_Sales', 'Global_Sales', 'Rank', 'Publisher_encoded']]

    X = df_encoded[features_to_keep]
    y = df_encoded['Global_Sales']

    # Store the exact column order from training data for consistent prediction input
    columns_when_trained = X.columns.tolist()

    return X, y

def show_model_diagnostics(y_test, y_pred):
    """
    Generates regression diagnostic plots using matplotlib and seaborn.
    Returns a matplotlib figure.
    """
    residuals = y_test - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

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
    return fig

def run_analysis_gradio(df_uploaded):
    """
    Runs the complete analysis pipeline including data cleaning, feature engineering,
    model training, and evaluation. Updates global state variables for the model and metrics.
    """
    global trained_model, scaler, model_coefficients, model_intercept, model_features, \
           global_r2_score, global_rmse_score, global_r2_adj_score

    if df_uploaded is None or df_uploaded.empty:
        return "Please upload a dataset in the 'Dataset Overview' tab first.", "", "", None, "", "", ""

    # Perform cleaning and get log message
    df_clean, cleaning_log_message = clean_data(df_uploaded.copy())

    if df_clean.empty:
        return "No data available for training after cleaning and feature engineering. Please check your dataset and filters.", "", "", None, "", "", ""

    X, y = feature_engineering(df_clean)

    if X.empty or y.empty:
        return "No data available for training after cleaning and feature engineering. Please check your dataset and filters.", "", "", None, "", "", ""

    # Split data - IMPORTANT: Split before scaling to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Scale numerical features AFTER train-test split to prevent data leakage ---
    num_features = ['NA_Sales', 'EU_Sales', 'Other_Sales', 'Year']

    # Initialize and fit scaler on training data only
    scaler = StandardScaler() # Re-initialize for a clean run
    train_num_features_exist = [col for col in num_features if col in X_train.columns]
    if train_num_features_exist:
        scaler.fit(X_train[train_num_features_exist])
        X_train[train_num_features_exist] = scaler.transform(X_train[train_num_features_exist])
        test_num_features_exist = [col for col in num_features if col in X_test.columns]
        X_test[test_num_features_exist] = scaler.transform(X_test[test_num_features_exist])
    else:
        cleaning_log_message += "\nWarning: No numerical features found to scale after cleaning. Check data or cleaning steps."

    # Train model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    trained_model = lr # Store the trained model globally

    # Make predictions
    y_pred = lr.predict(X_test)

    # Evaluation metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Calculate Adjusted R-squared
    n = X_test.shape[0]
    p = X_test.shape[1]
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else float('nan')

    # Store metrics globally
    global_r2_score = r2
    global_rmse_score = rmse
    global_r2_adj_score = r2_adj

    # Get top features (coefficients)
    coeff_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr.coef_
    }).sort_values('Coefficient', ascending=False)

    # Store coefficients and intercept for regression equation globally
    model_coefficients = lr.coef_
    model_intercept = lr.intercept_
    model_features = X.columns.tolist()

    # Generate equation string
    equation = f"Global_Sales = {model_intercept:.3f}"
    for i, feature in enumerate(model_features):
        coeff = model_coefficients[i]
        if coeff >= 0:
            equation += f" + {coeff:.3f} * {feature}"
        else:
            equation += f" - {abs(coeff):.3f} * {feature}"

    # Generate diagnostic plots
    diagnostics_plot = show_model_diagnostics(y_test, y_pred)

    # Prepare output strings for Gradio Markdown components
    metrics_output = f"""
    ### Model Performance Metrics
    * **R² Score (Coefficient of Determination)**: {r2:.3f}
    * **RMSE (Root Mean Squared Error)**: {rmse:.3f}
    * **Adjusted R² (Adjusted Coefficient of Determination)**: {r2_adj:.3f}
    """
    if np.isnan(r2_adj):
        metrics_output = metrics_output.replace(f"Adjusted R² (Adjusted Coefficient of Determination): {r2_adj:.3f}", "Adjusted R² (Adjusted Coefficient of Determination): N/A (Too few observations or features for calculation)")

    metrics_output += """
    * **R² Score**: Represents the proportion of the variance in the dependent variable (Global Sales) that is predictable from the independent variables (features). A value close to 1 indicates a good fit.
    * **RMSE**: Measures the average magnitude of the errors. It indicates how concentrated the data is around the line of best fit. Lower values are better.
    * **Adjusted R²**: A modified version of R² that has been adjusted for the number of predictors in the model. It increases only if the new term improves the model more than would be expected by chance, making it a better measure for comparing models with different numbers of features.
    """

    coeff_output = "### Top 10 Influential Features (Coefficients)\n"
    coeff_output += coeff_df.head(10).to_markdown(index=False)
    coeff_output += """
    * A **positive coefficient** means an increase in the feature's value leads to an increase in predicted global sales.
    * A **negative coefficient** means an increase in the feature's value leads to a decrease in predicted global sales.
    * Features related to regional sales (e.g., `NA_Sales`, `EU_Sales`, `Other_Sales`) often have the largest coefficients, indicating their strong direct relationship with `Global_Sales`.
    """

    regression_equation_output = f"### Regression Equation\n`{equation}`"
    regression_equation_output += """
    This equation represents the mathematical relationship learned by the model.
    Each coefficient reflects the impact of its corresponding feature on Global Sales.
    """

    multicollinearity_note = """
    ### Multicollinearity Note
    Multicollinearity occurs when independent variables in a regression model are correlated.
    While not explicitly checked with VIF (Variance Inflation Factor) in this app for simplicity,
    it's an important diagnostic. High multicollinearity can make coefficient interpretations
    less reliable and inflate their standard errors, though it doesn't typically affect the model's
    overall predictive power (R²) or the fitted values. Linear Regression handles multicollinearity
    implicitly by distributing the shared variance among correlated features.
    """
    status_message = "Analysis completed successfully! Model trained and evaluated."

    return cleaning_log_message, metrics_output, coeff_output, diagnostics_plot, regression_equation_output, multicollinearity_note, status_message

def predict_sales_gradio(year, platform, genre, na_sales, eu_sales, jp_sales, other_sales):
    """
    Makes a prediction using the trained model via Gradio inputs.
    """
    global trained_model, scaler, original_platforms, original_genres, columns_when_trained

    if trained_model is None or scaler is None or not hasattr(scaler, 'scale_'):
        return "Prediction failed: Model not trained or scaler not fitted. Please run 'Model Building' first."

    try:
        input_features = {
            'Year': year,
            'Platform': platform,
            'Genre': genre,
            'NA_Sales': na_sales,
            'EU_Sales': eu_sales,
            # JP_Sales is collected but not directly used in the model's features (X)
            'JP_Sales': jp_sales,
            'Other_Sales': other_sales,
        }

        # Create a DataFrame from the input features
        input_df = pd.DataFrame([input_features])

        # Apply one-hot encoding for Platform and Genre based on original categories
        # Ensure all possible one-hot encoded columns (from training) are present in the input_df
        # and set to 0 if the input category doesn't match
        for p_cat in original_platforms:
            input_df[f'Platform_{p_cat}'] = (input_df['Platform'] == p_cat).astype(int)
        for g_cat in original_genres:
            input_df[f'Genre_{g_cat}'] = (input_df['Genre'] == g_cat).astype(int)

        input_df = input_df.drop(columns=['Platform', 'Genre'])

        # Numerical features that were scaled during training
        num_features_to_scale = ['NA_Sales', 'EU_Sales', 'Other_Sales', 'Year']

        # Ensure all numerical features exist in input_df before scaling, add if missing with default 0.0
        for col in num_features_to_scale:
            if col not in input_df.columns:
                input_df[col] = 0.0

        # Scale the numerical features in the input DataFrame
        input_df[num_features_to_scale] = scaler.transform(input_df[num_features_to_scale])

        # Ensure that the input DataFrame has the exact same columns and order as the training data
        # Fill missing columns (e.g., one-hot encoded categories not present in current input) with 0
        final_input_df = pd.DataFrame(0, index=[0], columns=columns_when_trained)
        for col in columns_when_trained:
            if col in input_df.columns:
                final_input_df[col] = input_df[col].iloc[0] # Use .iloc[0] to get scalar value from single-row df

        prediction = trained_model.predict(final_input_df)
        return f"Predicted Global Sales: **{prediction[0]:.2f} million units**."

    except Exception as e:
        return f"Prediction failed: {str(e)}. This might happen if the model hasn't been trained yet, or if there's an issue with feature preparation. Please run 'Model Building' first."


# --- EDA Plotting Functions ---

def plot_sales_distribution(df):
    if df is None or df.empty:
        return None, "Please upload a dataset first."
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df['Global_Sales'], bins=50, kde=True, ax=ax1, color='lightcoral')
    ax1.set_title('Distribution of Global Sales (Millions)', fontsize=14)
    ax1.set_xlabel('Global Sales (Millions)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)

    sns.boxplot(x=df['Global_Sales'], ax=ax2, color='lightgreen')
    ax2.set_title('Boxplot of Global Sales (Millions)', fontsize=14)
    ax2.set_xlabel('Global Sales (Millions)', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    return fig, """
    **Key Insights:**
    1. Global sales distribution is **highly right-skewed**, indicating that most games have relatively low sales.
    2. The majority of games sell less than 1 million copies globally.
    3. There are many **outliers** with exceptionally high sales (e.g., Wii Sports), which will be addressed during data cleaning to improve model performance for the typical sales range.
    """

def plot_top_selling_games(df):
    if df is None or df.empty:
        return None, "Please upload a dataset first."
    top_games = df[['Name', 'Platform', 'Year', 'Global_Sales']].sort_values('Global_Sales', ascending=False).head(10)
    fig = plt.figure(figsize=(12, 7))
    sns.barplot(x='Global_Sales', y='Name', data=top_games, palette='viridis')
    plt.title('Top 10 Best-Selling Games Globally', fontsize=16)
    plt.xlabel('Global Sales (Millions)', fontsize=12)
    plt.ylabel('Game Title', fontsize=12)
    plt.grid(axis='x', linestyle=':', alpha=0.7)
    plt.tight_layout()
    return fig, """
    **Key Insights:**
    1. **Wii Sports** stands out significantly as the best-selling game.
    2. Many of the top-selling games are **Nintendo** exclusives or published by Nintendo, particularly for the Wii and NES platforms.
    3. The list highlights the long-term success of certain franchises like **Mario** and **Pokemon**.
    """

def plot_sales_by_platform(df):
    if df is None or df.empty:
        return None, "Please upload a dataset first."
    platform_sales = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(15)
    fig = plt.figure(figsize=(12, 7))
    sns.barplot(x=platform_sales.values, y=platform_sales.index, palette='crest')
    plt.title('Total Global Sales by Top 15 Platforms', fontsize=16)
    plt.xlabel('Total Global Sales (Millions)', fontsize=12)
    plt.ylabel('Platform', fontsize=12)
    plt.grid(axis='x', linestyle=':', alpha=0.7)
    plt.tight_layout()
    return fig, """
    **Key Insights:**
    1. **PS2** (PlayStation 2) and **X360** (Xbox 360) have the highest cumulative global sales, followed by **PS3** (PlayStation 3) and **Wii**.
    2. Older generation consoles like **NES** and **GB** (Game Boy) still hold significant total sales due to their longevity and large install bases.
    3. This data primarily reflects sales trends up to the dataset's cutoff year (around 2016), so newer platforms like PS4/Xbox One are not fully represented.
    """

def plot_sales_by_genre(df):
    if df is None or df.empty:
        return None, "Please upload a dataset first."
    genre_sales = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
    fig = plt.figure(figsize=(12, 7))
    sns.barplot(x=genre_sales.values, y=genre_sales.index, palette='magma')
    plt.title('Total Global Sales by Genre', fontsize=16)
    plt.xlabel('Total Global Sales (Millions)', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    plt.grid(axis='x', linestyle=':', alpha=0.7)
    plt.tight_layout()
    return fig, """
    **Key Insights:**
    1. **Action** and **Sports** are the dominant genres in terms of total global sales, indicating broad appeal.
    2. **Shooter** and **Role-Playing** games also perform exceptionally well.
    3. Genres like Puzzle and Strategy have lower overall sales, suggesting a more niche market or different monetization models not captured by pure sales figures.
    """

def plot_sales_over_time(df):
    if df is None or df.empty:
        return None, "Please upload a dataset first."
    # Ensure 'Year' is an integer before grouping, handle potential NaNs
    df_temp = df.dropna(subset=['Year']).copy()
    df_temp['Year'] = df_temp['Year'].astype(int)
    yearly_sales = df_temp.groupby('Year')['Global_Sales'].sum().sort_index()
    fig = plt.figure(figsize=(12, 7))
    sns.lineplot(x=yearly_sales.index, y=yearly_sales.values, marker='o', color='darkblue')
    plt.title('Global Video Game Sales Over Time (1980-2016)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Total Global Sales (Millions)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    return fig, """
    **Key Insights:**
    1. Sales grew steadily from the 1980s, experiencing a significant boom in the 2000s.
    2. Peak sales occurred around **2008-2009**, which coincides with the peak of the 7th generation console cycle (Wii, PS3, Xbox 360).
    3. A decline in sales is observed after 2009. This could be attributed to a shift towards digital distribution (not fully captured in this dataset) or changes in market dynamics. The data largely ends around 2016.
    """

def plot_correlation_matrix(df):
    if df is None or df.empty:
        return None, "Please upload a dataset first."
    sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    corr_matrix = df[sales_cols].corr()
    fig = plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Matrix of Regional and Global Sales', fontsize=16)
    plt.tight_layout()
    return fig, """
    **Key Insights:**
    1. **NA_Sales (North America)** shows the strongest positive correlation with `Global_Sales` (0.94), indicating it's the primary driver of worldwide sales in this dataset.
    2. **EU_Sales (Europe)** also has a very strong positive correlation (0.90) with `Global_Sales`.
    3. **JP_Sales (Japan)** exhibits a weaker, but still positive, correlation (0.62) compared to NA and EU sales, suggesting the Japanese market has distinct sales patterns or a smaller overall contribution to global figures.
    4. `Other_Sales` shows a moderate correlation (0.75).
    These insights support the decision to use regional sales (excluding JP_Sales for model simplicity) as strong predictors for Global Sales.
    """

# --- Gradio Interface Definition ---

with gr.Blocks(title="Video Game Sales Analysis & Prediction") as demo:
    gr.Markdown("# 🎮 Video Game Sales Analysis & Prediction")
    gr.Markdown("""
    This app performs an in-depth analysis of video game sales data and utilizes a Linear Regression model
    to predict global sales based on various features like platform, genre, and regional sales.
    """)

    # A Gradio State component to hold the DataFrame across tabs.
    # This is crucial because Gradio functions are stateless by default.
    df_state = gr.State(value=pd.DataFrame())

    with gr.Tab("Dataset Overview") as dataset_overview_tab:
        gr.Markdown("### Raw Dataset Information")
        file_input = gr.File(label="Upload vgsales.csv", file_types=[".csv"])
        
        with gr.Row(): # Group buttons in a row
            load_button = gr.Button("Load Data")
            update_button = gr.Button("Update Data") # New Update button
            
        dataset_info_output = gr.Markdown("Please upload a CSV file to see dataset information.")

        # When load_button is clicked, call load_data_gradio and update df_state and dataset_info_output
        load_button.click(
            load_data_gradio,
            inputs=[file_input],
            outputs=[df_state, dataset_info_output]
        )
        
        # When update_button is clicked, also call load_data_gradio
        update_button.click(
            load_data_gradio,
            inputs=[file_input],
            outputs=[df_state, dataset_info_output]
        )

    with gr.Tab("Exploratory Data Analysis") as eda_tab: # Added as eda_tab
        gr.Markdown("### Exploratory Data Analysis (EDA)")
        gr.Markdown("Dive into the data to uncover patterns, anomalies, trends, and relationships. Visualizations help in understanding the distribution and interdependencies of features.")

        eda_options = [
            "Sales Distribution",
            "Top Selling Games",
            "Sales by Platform",
            "Sales by Genre",
            "Sales Over Time",
            "Correlation Matrix"
        ]
        eda_dropdown = gr.Dropdown(eda_options, label="Select visualization", value="Sales Distribution")
        eda_plot_output = gr.Plot(label="Plot")
        eda_insights_output = gr.Markdown("Insights from the plot will appear here.")

        # Function to update EDA plots based on dropdown selection
        def update_eda_plot_wrapper(choice, current_df):
            if current_df.empty:
                return None, "Please upload a dataset in the 'Dataset Overview' tab first."
            if choice == "Sales Distribution":
                return plot_sales_distribution(current_df)
            elif choice == "Top Selling Games":
                return plot_top_selling_games(current_df)
            elif choice == "Sales by Platform":
                return plot_sales_by_platform(current_df)
            elif choice == "Sales by Genre":
                return plot_sales_by_genre(current_df)
            elif choice == "Sales Over Time":
                return plot_sales_over_time(current_df)
            elif choice == "Correlation Matrix":
                return plot_correlation_matrix(current_df)
            return None, "Invalid selection."

        # Trigger update when dropdown changes
        eda_dropdown.change(
            update_eda_plot_wrapper,
            inputs=[eda_dropdown, df_state], # Pass the df_state
            outputs=[eda_plot_output, eda_insights_output]
        )
        
        # Trigger initial plot when EDA tab is selected
        eda_tab.select( # Use .select() for tab changes
            lambda df: update_eda_plot_wrapper("Sales Distribution", df),
            inputs=[df_state],
            outputs=[eda_plot_output, eda_insights_output]
        )


    with gr.Tab("Model Building"):
        gr.Markdown("### Model Building")
        gr.Markdown("""
        This section builds and evaluates a Linear Regression model to predict `Global_Sales`.
        The pipeline includes data cleaning, feature engineering (one-hot encoding, feature selection),
        data splitting, scaling, model training, and evaluation.
        """)
        run_analysis_button = gr.Button("Run Full Analysis")
        analysis_status_message = gr.Markdown("")
        cleaning_log_markdown = gr.Markdown("Data cleaning log will appear here.")
        metrics_markdown = gr.Markdown("Model performance metrics will appear here.")
        coefficients_markdown = gr.Markdown("Most influential features (coefficients) will appear here.")
        diagnostics_plot = gr.Plot(label="Regression Diagnostic Plots")
        equation_markdown = gr.Markdown("Regression equation will appear here.")
        multicollinearity_markdown = gr.Markdown("Note on multicollinearity will appear here.")

        run_analysis_button.click(
            run_analysis_gradio,
            inputs=[df_state], # Pass the df_state to the function
            outputs=[
                cleaning_log_markdown,
                metrics_markdown,
                coefficients_markdown,
                diagnostics_plot,
                equation_markdown,
                multicollinearity_markdown,
                analysis_status_message
            ]
        )

    with gr.Tab("Make Predictions"):
        gr.Markdown("### Predict Global Sales for a New Game")
        gr.Markdown("""
        Use the trained model to predict the global sales of a new video game.
        Fill in the details below and click 'Predict Global Sales'.
        """)

        # Helper functions to get dynamic choices for dropdowns
        def get_platform_choices_for_dropdown():
            return gr.Dropdown(choices=original_platforms if original_platforms else [], label="Platform", info="Select the gaming platform.", interactive=True)
        def get_genre_choices_for_dropdown():
            return gr.Dropdown(choices=original_genres if original_genres else [], label="Genre", info="Select the game genre.", interactive=True)

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Game Details")
                year_input = gr.Number(label="Release Year", minimum=1980, maximum=2023, value=2010, step=1, info="The year the game was released (e.g., 2010).")
                # Dropdowns initialized empty, updated dynamically
                platform_dropdown = gr.Dropdown(choices=[], label="Platform", info="Select the gaming platform.", interactive=True)
                genre_dropdown = gr.Dropdown(choices=[], label="Genre", info="Select the game genre.", interactive=True)
            with gr.Column():
                gr.Markdown("#### Regional Sales (in millions of units)")
                na_sales_input = gr.Number(label="North America Sales", minimum=0.0, value=1.0, step=0.01, info="Sales in North America (e.g., 1.0 for 1 million units).")
                eu_sales_input = gr.Number(label="Europe Sales", minimum=0.0, value=0.5, step=0.01, info="Sales in Europe (e.g., 0.5 for 0.5 million units).")
                jp_sales_input = gr.Number(label="Japan Sales", minimum=0.0, value=0.3, step=0.01, info="Sales in Japan (e.g., 0.3 for 0.3 million units). Note: JP_Sales is collected but not directly used as a predictor in this model for simplicity.")
                other_sales_input = gr.Number(label="Other Regions Sales", minimum=0.0, value=0.2, step=0.01, info="Sales in other regions of the world (e.g., 0.2 for 0.2 million units).")

        predict_button = gr.Button("Predict Global Sales")
        prediction_output = gr.Markdown("Predicted Global Sales will appear here.")

        predict_button.click(
            predict_sales_gradio,
            inputs=[
                year_input,
                platform_dropdown,
                genre_dropdown,
                na_sales_input,
                eu_sales_input,
                jp_sales_input, # Still pass, even if not directly used in model, for interface consistency
                other_sales_input
            ],
            outputs=prediction_output
        )

        # Update dropdown choices when the "Model Building" button is clicked (after model is trained)
        run_analysis_button.click(
            lambda: [get_platform_choices_for_dropdown(), get_genre_choices_for_dropdown()],
            inputs=[],
            outputs=[platform_dropdown, genre_dropdown]
        )
        # Also, update dropdowns when the "Make Predictions" tab is loaded to ensure they have values
        demo.load(
            lambda: [get_platform_choices_for_dropdown(), get_genre_choices_for_dropdown()],
            outputs=[platform_dropdown, genre_dropdown]
        )


    with gr.Tab("Results Summary"):
        gr.Markdown("### Project Results Summary")
        gr.Markdown("""
        This section provides a concise summary of the project's key findings, model performance, and insights.
        """)
        summary_markdown = gr.Markdown("Run the 'Model Building' analysis to see the results summary.")

        def get_summary_gradio():
            # Access global variables for metrics and other info
            if trained_model is None:
                return "Model not trained yet. Please run 'Model Building' first."

            summary_text = "## Final Project Summary\n\n"
            summary_text += "**Project Goal:**\n"
            summary_text += """
            The primary objective of this project is to analyze historical video game sales data and develop a Linear Regression model capable of predicting global sales. Furthermore, the project aims to identify key factors influencing game sales and deploy the model through a user-friendly Streamlit (now Gradio) application.
            """

            summary_text += "\n**Key Findings & Model Performance:**\n"
            if not np.isnan(global_r2_score):
                summary_text += f"1. **Model Performance:** The Linear Regression model achieved an **R² score (Coefficient of Determination) of {global_r2_score:.3f}** on the test data.\n"
                summary_text += f"   The **RMSE (Root Mean Squared Error) is {global_rmse_score:.3f}** million units, indicating the typical error in predictions.\n"
                if not np.isnan(global_r2_adj_score):
                     summary_text += f"   The **Adjusted R² (Adjusted Coefficient of Determination) is {global_r2_adj_score:.3f}**, providing a more robust measure of fit considering the number of features."
                else:
                    summary_text += "   Adjusted R² (Adjusted Coefficient of Determination): N/A (e.g., too few observations or features for calculation)."

            else:
                summary_text += "Model performance metrics are not available. Please run 'Model Building' first to populate them.\n"


            summary_text += """
            2. **Strongest Predictors:** Regional sales, particularly **North America Sales (`NA_Sales`)** and **Europe Sales (`EU_Sales`)**, were identified as the strongest predictors of global success, showing very high correlations with `Global_Sales`. `Other_Sales` also contributed significantly.
            3. **Genre Impact:** **Action** and **Sports** genres consistently lead in total global sales, indicating their widespread market appeal.
            4. **Platform Dominance:** Historically, platforms like **PS2**, **X360**, and **Wii** have contributed significantly to overall sales.
            5. **Temporal Trends:** The video game industry saw a peak in sales around **2008-2009**, followed by a decline in subsequent years (likely due to market shifts and increased digital distribution, which is not fully captured by this dataset).

            **Model Strengths:**
            - **Interpretability:** Linear Regression offers high interpretability, allowing us to understand the direct impact (coefficients) of each feature on predicted sales.
            - **Predictive Power:** The R² score indicates a reasonable ability to explain the variance in global sales.
            - **Robust Preprocessing:** The pipeline includes robust data cleaning (missing values, outliers) and feature engineering (one-hot encoding, scaling after train-test split to prevent data leakage).
            - **Deployment Ready:** Successfully deployed via Streamlit (and now Gradio), providing an interactive and user-friendly interface for analysis and prediction.

            **Limitations and Potential Improvements:**
            1.  **Data Recency:** The dataset covers most sales up to 2016. Including more recent data (post-2020) would significantly enhance the model's relevance to current market trends, especially with the rise of digital sales, mobile gaming, and newer console generations.
            2.  **Model Complexity:** While Linear Regression is good for interpretability, it assumes a linear relationship. Exploring **non-linear models** (e.g., Polynomial Regression, Ridge/Lasso Regression for regularization, or tree-based models like Random Forest/XGBoost) could capture more complex patterns and potentially improve prediction accuracy.
            3.  **Additional Features:** The model could be enhanced by incorporating external features such as:
                * **Review scores/ratings** (e.g., Metacritic scores)
                * **Marketing budget** and promotional spending
                * **Developer/Publisher reputation** (beyond simple encoding)
                * **Online/Multiplayer features** (if quantifiable)
                * **Game engine/technology**
            4.  **Advanced Feature Engineering:** More advanced feature engineering, such as creating **interaction terms** between existing features (e.g., Genre × Platform sales effect) or temporal features (e.g., game's age), could lead to better insights.
            5.  **Hyperparameter Tuning:** While Linear Regression has few hyperparameters, for more complex models, systematic **hyperparameter tuning** (e.g., GridSearchCV, RandomizedSearchCV) would be crucial to find optimal model configurations. For this simple linear regression, default parameters generally work well.
            6.  **Multicollinearity Handling:** Although noted, a more rigorous check using VIF (Variance Inflation Factor) and removal or merging of highly correlated features could be explored if highly correlated features are causing issues with coefficient interpretability.
            """

            summary_text += "\n### Final Regression Equation\n"
            if model_intercept is not None and model_coefficients is not None and model_features:
                equation = f"Global_Sales = {model_intercept:.3f}"
                for i, feature in enumerate(model_features):
                    coeff = model_coefficients[i]
                    if coeff >= 0:
                        equation += f" + {coeff:.3f} * {feature}"
                    else:
                        equation += f" - {abs(coeff):.3f} * {feature}"
                summary_text += f"```python\n{equation}\n```"
            else:
                summary_text += "Regression equation is not available. Please run 'Model Building' first."

            return summary_text

        # Update summary when "Model Building" is run
        run_analysis_button.click(
            get_summary_gradio,
            inputs=[],
            outputs=summary_markdown
        )
        # Also, update summary when the tab is selected if a model already exists
        demo.load(
            get_summary_gradio,
            outputs=summary_markdown
        )


demo.launch()
