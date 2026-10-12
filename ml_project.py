# =====================================================================
# IMPORTING LIBRARIES
# =====================================================================

# Imports Pandas and nicknames it 'pd' to save typing later. 
# Used for data manipulation, cleaning, and creating DataFrames (tables).
import pandas as pd

# Imports NumPy and nicknames it 'np'. 
# Used for fast, high-performance mathematical and numerical array operations.
import numpy as np

# Imports Scikit-learn (sklearn), the primary machine learning library.
# Contains built-in tools for modeling, preprocessing, and testing datasets.
import sklearn

# ==============================================================================
# DATA VISUALIZATION ENGINE IMPORT
# ==============================================================================

# Imports the plotting framework from the Matplotlib library.
# We assign it the standard industry nickname 'plt' to keep our charting code 
# clean and readable when drawing graphs, titles, and labels.
import matplotlib.pyplot as plt


# =====================================================================
# ENVIRONMENT VERIFICATION
# =====================================================================

# Uses Python f-strings to dynamically look up and print the installed versions
# of the packages from your specific virtual environment (.venv).
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print("Setup successful!")


# =====================================================================
# LOADING DATA
# =====================================================================

# Targets a built-in dataset containing real estate data from California.
# This prevents you from needing to download a separate CSV file to get started.
from sklearn.datasets import fetch_california_housing

# Fetches the housing data. 
# Setting 'as_frame=True' tells Python to structure it immediately into a clean table structure.
housing = fetch_california_housing(as_frame=True)

# Extracts the complete table (containing both the housing features and the house price target column).
df = housing.frame


# =====================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================

# '.shape' outputs a tuple showing the total size of your table format: (Rows, Columns).
print(f"Dataset shape: {df.shape}")

# Prints the names of the column metrics gathered (e.g., Median Income, House Age, Rooms).
print(f"Features: {list(housing.feature_names)}")

print("\nFirst few rows:")
# '.head()' is a preview function that returns just the first 5 rows of data to see how it looks.
print(df.head())


# =====================================================================
# DATA CLEANING & QUALITY CHECKS
# =====================================================================

# '.describe()' automatically calculates a statistical summary of every column:
# counting rows, finding averages (mean), standard deviation, minimums, maximums, and percentiles.
print(df.describe())

# Check for missing data values.
# '.isnull()' flags every empty cell, the first '.sum()' counts them by column, 
# and the second '.sum()' totals them up to ensure there are zero broken missing fields.
print(f"\nMissing values: {df.isnull().sum().sum()}")

# =====================================================================
# IMPORTING THE SPLITTING TOOL
# =====================================================================

# Imports the train_test_split function from Scikit-learn.
# This function shuffles and splits your dataset automatically so you don't have to do it manually.
from sklearn.model_selection import train_test_split


# =====================================================================
# ISOLATING THE QUESTIONS FROM THE ANSWERS
# =====================================================================

# Creates 'X' (uppercase by convention for matrices/tables).
# This drops the 'MedHouseVal' column, leaving only the clues/features (like income, age, rooms) used to make predictions.
# 'axis=1' tells Python to drop a vertical *column* rather than a horizontal row.
X = df.drop('MedHouseVal', axis=1)

# Creates 'y' (lowercase by convention for a single vector/list).
# This isolates *only* the 'MedHouseVal' (Median House Value) column. 
# This is the "target"—the exact answer key the AI will try to guess.
y = df['MedHouseVal']


# =====================================================================
# THE 80/20 SPLIT SANITIZATION
# =====================================================================

# Splits your data into 4 distinct piles at the exact same time:
#   - X_train / y_train: The practice test your AI will study from (80% of data).
#   - X_test / y_test: The final exam used to grade how smart your AI actually is (20% of data).
# 'test_size=0.2' reserves exactly 20% of your data for testing.
# 'random_state=42' locks in the shuffle seed. This ensures that every time you run this script, 
# it shuffles the rows the exact same way, keeping your numbers perfectly reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =====================================================================
# VERIFYING THE SPLIT RATIOS
# =====================================================================

# 'X_train.shape[0]' targets just index 0 of the shape tuple, which is the exact row count.
# This prints out the total number of practice houses the model gets to study.
print(f"Training samples: {X_train.shape[0]}")

# This prints out the total number of unseen houses hidden away for grading the model later.
print(f"Test samples: {X_test.shape[0]}")

# =====================================================================
# IMPORTING THE ALGORITHM
# =====================================================================

# Imports LinearRegression from Scikit-learn.
# This is a classic machine learning algorithm that finds a straight line of best fit 
# through your data points to make future predictions.
from sklearn.linear_model import LinearRegression


# =====================================================================
# CREATING THE MACHINE LEARNING MODEL
# =====================================================================

# 1. Create the estimator
# This creates a blank, untrained instance of the model and names it 'lin_reg'.
# Think of this as opening a fresh, empty brain that doesn't know anything about real estate yet.
lin_reg = LinearRegression()


# =====================================================================
# TRAINING THE MODEL (THE "STUDY" PHASE)
# =====================================================================

# 2. Fit it to the training data
# '.fit()' is the universal Scikit-learn command that kicks off the actual training process.
# We hand it the clues (X_train) and the true answers (y_train). The model analyzes all 16,000+ 
# houses, learns the patterns, and writes its own complex mathematical formula.
lin_reg.fit(X_train, y_train)


# =====================================================================
# TAKING THE FINAL EXAM (THE "PREDICT" PHASE)
# =====================================================================

# 3. Predict on data it has not seen
# We hand the model *only* the hidden clues (X_test). We strictly do not show it the answers.
# The model uses its newly invented formula to calculate its best guess for every single house.
# All of these AI guesses are saved into a new list called 'y_pred_lin'.
y_pred_lin = lin_reg.predict(X_test)


# =====================================================================
# GRADING THE EXAM (VISUAL INSPECTION)
# =====================================================================

print("First five predictions vs actual values:")

# 'zip()' acts like a physical zipper, locking the AI's guesses and the real answers together row-by-row.
# '[:5]' ensures we only look at the first 5 houses so we don't flood the terminal screen.
for predicted, actual in zip(y_pred_lin[:5], y_test[:5]):
    
    # ':.2f' is a formatting trick that rounds the numbers to exactly 2 decimal places.
    # This prints out a side-by-side comparison so you can see how close the AI got to the real truth.
    print(f"Predicted: {predicted:.2f}   Actual: {actual:.2f}")

# =====================================================================
# PRINTING THE HEADER BLOCK
# =====================================================================

# Prints the section title. The '\n' injects a clean blank line right before it 
# to keep our terminal screen organized and easy to read.
print("\nWhat the model learned:")


# =====================================================================
# PAIRING FEATURES WITH THEIR INVENTED MATHEMATICAL WEIGHTS
# =====================================================================

# 'X_train.columns' holds the list of clue names (like MedInc, HouseAge, Rooms).
# 'lin_reg.coef_' holds the "coefficients"—the exact mathematical weights the AI invented.
# 'zip()' pairs them up row-by-row (e.g., matching the clue 'MedInc' to its specific weight value).
for feature, coef in zip(X_train.columns, lin_reg.coef_):
    
    # =====================================================================
    # FORMATTING AND OUTPUTTING THE RESULTS
    # =====================================================================
    
    # This line prints out each clue alongside its weight using two special formatting tricks:
    # 
    # '{feature:>10}' -> Right-aligns the clue text to fit a 10-character wide block. 
    #                    This lines up all the colons (:) in a perfectly straight vertical line.
    #
    # '{coef:+.4f}'   -> Forces a '+' or '-' sign to always show up in front of the number, 
    #                    and rounds the decimal calculation to exactly 4 decimal places.
    print(f"{feature:>10}: {coef:+.4f}")

    # =====================================================================
# IMPORTING THE EVALUATION AND BASELINE TOOLS
# =====================================================================

# Imports evaluation tools from Scikit-learn:
#   - mean_absolute_error (MAE): Measures the average absolute dollar mistake size.
#   - mean_squared_error (MSE): Used to compute the Root Mean Squared Error.
#   - r2_score (R²): Calculates the percentage score of patterns captured.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Imports DummyRegressor. This is a deliberately "dumb" model used solely as a baseline.
# If a real machine learning model cannot beat this dummy model, your AI is useless.
from sklearn.dummy import DummyRegressor


# =====================================================================
# CREATING A REUSABLE REPORT CARD FUNCTION
# =====================================================================

# Defines a custom function named 'evaluate' so we don't have to repeat code.
# It requires three inputs: a label name, the real answers (y_true), and the AI's guesses (y_pred).
def evaluate(name, y_true, y_pred):
    
    # 1. Calculate Mean Absolute Error (MAE). 
    # This tells us the average distance between the guesses and the actual values.
    mae = mean_absolute_error(y_true, y_pred)
    
    # 2. Calculate Root Mean Squared Error (RMSE). 
    # 'np.sqrt()' takes the square root of the MSE. This puts the squared error back 
    # into the original units (dollars), heavily emphasizing if the model made massive errors.
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 3. Calculate the R-squared (R²) score to check the final accuracy percentage.
    r2 = r2_score(y_true, y_pred)
    
    # 4. Print out the formatted report card to the terminal screen.
    print(f"{name}")
    
    # Note on the $ calculations below: The California housing dataset tracks prices in hundreds of thousands 
    # of dollars (e.g., a value of 2.0 means $200,000). Multiplying by 100,000 converts it back to real dollars.
    # ':,.0f' formats the number with comma separators (like $120,000) and rounds off any pennies.
    print(f"  MAE:  {mae:.3f}  (about ${mae * 100000:,.0f})")
    print(f"  RMSE: {rmse:.3f}  (about ${rmse * 100000:,.0f})")
    print(f"  R2:   {r2:.3f}")


# =====================================================================
# BUILDING AND TRAINING THE BASELINE DUMMY
# =====================================================================

# 'strategy="mean"' tells this dummy model to completely ignore all clues (X_train).
# It simply calculates the average house price of the dataset and will guess that exact same number every single time.
dummy = DummyRegressor(strategy="mean")

# Feeds the data to the dummy model so it can calculate what that overall average price is.
dummy.fit(X_train, y_train)


# =====================================================================
# RUNNING THE SIDE-BY-SIDE COMPETITION
# =====================================================================

# Generates predictions using the dummy model on the test data and sends them to our report card function.
# This establishes the lowest acceptable score (the baseline).
evaluate("Always predict the average", y_test, dummy.predict(X_test))

# Sends our earlier Linear Regression AI guesses to the exact same report card function.
# This lets us directly see how much smarter our real AI is compared to the dummy average strategy.
evaluate("Linear regression", y_test, y_pred_lin)

# ==============================================================================
# BUILDING A PIPELINE
# ==============================================================================

# # Imports the K-Nearest Neighbors regression model from Scikit-learn.
# This algorithm makes predictions by locating the most visually or numerically 
# similar historic districts based on spatial distance calculations.
from sklearn.neighbors import KNeighborsRegressor

# Imports the Pipeline tool from Scikit-learn.
# This allows us to chain multiple preprocessing steps and a final model together 
# so they behave cleanly as a single unified object.
from sklearn.pipeline import Pipeline

# Imports the StandardScaler preprocessing tool.
# This scales data so that features with massive ranges don't override small ones.
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------------------
# KNN WITHOUT SCALING
# ------------------------------------------------------------------------------

# Creates an unscaled instance of the KNN algorithm looking at the 10 closest neighbors.
# WITHOUT scaling, features with huge values (like Population in the thousands) 
# completely drown out small values (like MedInc which stays under 16).
knn = KNeighborsRegressor(n_neighbors=10)

# Fits the unscaled model directly to our training data clues and answers.
knn.fit(X_train, y_train)

# Evaluates the unscaled KNN guesses using your custom report card function.
# Because the feature scales are broken, this will yield poor results (R² ~ 0.155).
evaluate("KNN without scaling", y_test, knn.predict(X_test))


# ------------------------------------------------------------------------------
# KNN WITH SCALING (USING A PIPELINE)
# ------------------------------------------------------------------------------

# Creates a pipeline that combines scaling and modeling into one single object.
# This strictly prevents "Data Leakage" because the scaler only calculates its 
# math settings using 'X_train' during fit, keeping 'X_test' perfectly hidden.
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsRegressor(n_neighbors=10)),
])

# Trains the entire pipeline. Python automatically standardizes the features 
# first, and then immediately feeds those scaled values into the KNN algorithm.
knn_pipeline.fit(X_train, y_train)

# Evaluates the pipeline's performance on the test data.
# Notice how scaling drastically improves accuracy metrics (R² jumps to ~ 0.676).
evaluate("KNN with scaling (pipeline)", y_test, knn_pipeline.predict(X_test))


# ------------------------------------------------------------------------------
# TESTING IF SCALING HELPS LINEAR REGRESSION
# ------------------------------------------------------------------------------

# Creates a matching pipeline structure for our Linear Regression baseline model.
lin_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])

# Trains the scaled linear regression pipeline on your dataset.
lin_pipeline.fit(X_train, y_train)

# Evaluates the scaled linear regression model performance metrics.
# Algebraic models like Linear Regression adjust their math weights to any scale, 
# meaning these results will be perfectly identical to your unscaled baseline.
evaluate("Linear regression (pipeline)", y_test, lin_pipeline.predict(X_test))

# ==============================================================================
# CROSS-VALIDATION & MULTI-MODEL BENCHMARKING
# ==============================================================================
# Context: Evaluating models on a single test split can lead to overoptimistic
# or volatile metrics. Cross-validation solves this by cycling through data.
# ==============================================================================

# Imports KFold to handle split mechanics and cross_val_score to automate loops.
from sklearn.model_selection import KFold, cross_val_score

# Imports a Single Decision Tree model (vulnerable to high variance/overfitting).
from sklearn.tree import DecisionTreeRegressor

# Imports an Ensemble Bagging model (combines 100 trees to stabilize variance).
from sklearn.ensemble import RandomForestRegressor


# --- CONSTRUCTING THE EXPERIMENT WORKBENCH ---
# We bundle our custom pipelines and brand new tree-based models into a Python 
# dictionary. This allows us to systematically benchmark them side-by-side.
models = {
    # Reuses the standard pipelines to shield against leakage.
    "Linear regression": lin_pipeline,
    "KNN (scaled)": knn_pipeline,
    
    # Decision Trees don't need scaling because they split data using boundaries,
    # rather than calculating geometric feature distances or algebraic weights.
    "Decision tree": DecisionTreeRegressor(random_state=42),
    
    # Random Forest is a collections of trees. 'n_jobs=-1' tells Python to use
    # ALL available CPU processor cores in parallel to maximize training speed.
    "Random forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
}


# --- SETTING UP THE K-FOLD SPLITTING MECHANICS ---
# KFold shuffles the training data and chunks it into 5 unique, equal-sized folds.
# The algorithm will train 5 separate times. In each iteration, 4 folds act as the 
# study material, and 1 fold is held back completely blind to serve as the exam.
kfold = KFold(n_splits=5, shuffle=True, random_state=42)


# --- EXECUTING THE COMPARATIVE BENCHMARK ---
print("5-fold cross-validation RMSE (lower is better):")

for name, model in models.items():
    # 'cross_val_score' automatically runs the split-fit-score loop 5 times.
    # CRITICAL MATH NOTE: Scikit-learn requires all evaluation scores to follow 
    # a "higher is better" rule. To fit this rule, it outputs a NEGATIVE RMSE.
    scores = cross_val_score(
        model, X_train, y_train,
        cv=kfold, scoring="neg_root_mean_squared_error"
    )
    
    # We flip the negative sign back to positive to get the true, real-world RMSE error.
    rmse_scores = -scores
    
    # Outputs a perfectly aligned comparison board using formatting tricks:
    #   1. '{name:>18}' -> Right-aligns the model labels to a 18-character wide block.
    #   2. '.mean()'    -> Calculates the average overall error of the 5 runs.
    #   3. '.std()'     -> Measures volatility (how much the error swung between folds).
    print(f"{name:>18}: {rmse_scores.mean():.3f}  (+/- {rmse_scores.std():.3f})")


# ==============================================================================
# DIAGNOSING OVERFITTING IN DECISION TREES
# ==============================================================================
# Context: A single Decision Tree can ask infinitely deep questions, allowing
# it to map patterns perfectly—including random noise in the training set.
# ==============================================================================

# Imports the mathematical squared error metric required to calculate variance.

# Instantiates a baseline Decision Tree Regressor.
# By leaving parameters like 'max_depth' empty, the tree is allowed to grow 
# completely unchecked until every single training house resides in its own leaf.
tree = DecisionTreeRegressor(random_state=42)

# Fits the tree model exclusively to your training clues and target answers.
tree.fit(X_train, y_train)


# --- CALCULATING SELF-TRAINING ERROR ---
# We make predictions on the SAME exact data the model just studied ('X_train').
# Because a deep tree can build a unique rule for every single house row, 
# its predictions on its own training set will be near-flawless (RMSE near 0.000).
train_rmse = np.sqrt(mean_squared_error(y_train, tree.predict(X_train)))


# --- PRINTING THE SELF-DIAGNOSIS ---
# This result acts as a diagnostic warning flag. If your training RMSE is 0.000
# but your Cross-Validation RMSE is much higher (~0.733), 
# it proves the model has severely OVERFIT and memorized the data noise.
print(f"Decision tree RMSE on its own training data: {train_rmse:.3f}")

# ==============================================================================
# HYPERPARAMETER TUNING: DIAGNOSING UNDERFITTING VS. OVERFITTING
# ==============================================================================
# This loop evaluates how changing a tree's maximum depth affects its ability 
# to generalize to unseen data, highlighting the bias-variance tradeoff.
# ==============================================================================

# Imports cross_validate to evaluate performance on BOTH training and validation
# folds concurrently during the K-Fold process.
from sklearn.model_selection import cross_validate

# Prints a formatted column header to cleanly organize table results in PowerShell.
print("\nmax_depth | train RMSE | validation RMSE")

# Iterates through different depth constraints. 
# 'None' means unlimited growth, allowing the tree to grow until all leaves are pure.
for depth in [2, 4, 6, 8, 10, 15, None]:
    
    # Executes 5-fold cross-validation on a decision tree restricted by the current depth.
    results = cross_validate(
        DecisionTreeRegressor(max_depth=depth, random_state=42),
        X_train, 
        y_train,
        cv=kfold,                       # Uses the identical 5-fold split as before
        scoring="neg_root_mean_squared_error",  # Metric target (Negative RMSE)
        return_train_score=True,        # CRITICAL: Tracks training fold scores to diagnose overfitting
    )
    
    # --- MATH ADJUSTMENT ---
    # Scikit-learn outputs negative scores to follow its "higher is better" rule.
    # We negate them (-) to convert back into positive, standard RMSE error values.
    # .mean() averages the performance metrics across all 5 cross-validation iterations.
    train = -results["train_score"].mean()
    valid = -results["test_score"].mean()
    
    # --- OUTPUT FORMATTING ---
    # Python formatting syntax ensures that output lines align into straight vertical columns:
    #   - {str(depth):>9}  : Places the depth string right-aligned in a 9-character block.
    #   - {train:10.3f}    : Formats train error to 3 decimal places in a 10-character block.
    #   - {valid:15.3f}    : Formats validation error to 3 decimal places in a 15-character block.
    print(f"{str(depth):>9} | {train:10.3f} | {valid:15.3f}")

# ==============================================================================
# TRAIN THE FINAL CHAMPION MODEL AND CONCLUDE TESTING
# ==============================================================================
# Context: Now that cross-validation has proven Random Forest to be our absolute 
# best predictor (lowest error), we deploy it for the final real-world check.
# ==============================================================================

# --- INITIALIZE THE CHAMPION CONFIGURATION ---
# We recreate a fresh instance of the winning Random Forest model configuration.
#   1. 'n_estimators=100'  -> Spawns a crowd of 100 individual decision trees.
#   2. 'random_state=42'   -> Locks the randomness seed for repeatable results.
#   3. 'n_jobs=-1'         -> Forces Python to utilize all multi-core processing
#                             power available on your computer CPU for maximum speed.
final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)


# --- COMPLETE THE TRAINING LIFECYCLE ---
# We fit the final model to the complete training set ('X_train' and 'y_train'). 
# This allows the forest to study every single training point available all at once, 
# ensuring its decision patterns are as deep and accurate as possible.
final_model.fit(X_train, y_train)


# --- THE BLIND TESTING EXAM (THE REVEAL) ---
# CRITICAL MACHINE LEARNING PROTOCOL: This is the ONLY time we ever touch our test data.
# The 'X_test' set has been perfectly hidden until now, acting as a blind final exam. 
#
#   1. final_model.predict(X_test) -> Generates real, blind housing predictions.
#   2. y_test                      -> The hidden answer key sheet.
#   3. evaluate()                  -> Your custom printing tool that spits out the
#                                     final MAE, RMSE, and R² scores.
evaluate("Final model: random forest (test set)", y_test, final_model.predict(X_test))

# ==============================================================================
# GOLDEN RULE REMINDER:
# If you reuse or change your models after seeing this score, you risk leaking
# test data patterns into your decisions.
# ==============================================================================


# ==============================================================================
# EXTRACTING & VISUALIZING FEATURE IMPORTANCES
# ==============================================================================
# Context: Random Forests act as a "black box." To explain how the model makes
# decisions, we extract its internal split metrics to rank which columns matter.
# ==============================================================================

# --- EXTRACT AND RANK THE FEATURES ---
# 1. final_model.feature_importances_ -> Retrieves an array of mathematical weights 
#    representing how much each column decreased variance during training.
# 2. index=X_train.columns -> Maps those raw numeric weights back to the text names.
# 3. pd.Series(...) -> Converts the raw array into a structured Pandas Series object.
# 4. .sort_values(ascending=False) -> Sorts the rankings from highest to lowest.
importances = pd.Series(
    final_model.feature_importances_, index=X_train.columns
).sort_values(ascending=False)


# --- PRINT THE RANKINGS TO THE CONSOLE ---
print("\nFeature importance:")

# Iterates through the sorted pandas series to display a clean text leaderboard.
for feature, importance in importances.items():
    # '{feature:>10}' right-aligns column labels inside a 10-character wide block.
    # '{importance:.3f}' limits the importance decimal value to 3 precise places.
    print(f"{feature:>10}: {importance:.3f}")


# --- CONSTRUCT THE HORIZONTAL BAR CHART ---
# 1. importances.sort_values() -> Flips the data back so the biggest bar sits at the top.
# 2. kind="barh" -> Forces the plot to draw horizontal bars (easier to read feature names).
# 3. color="steelblue" -> Applies a clean, professional blue palette to the visualization.
importances.sort_values().plot(kind="barh", color="steelblue")

# Sets the main context title at the very top of the graphic window canvas.
plt.title("Which features drive the predictions?")

# Labels the horizontal x-axis so stakeholders understand the scale metrics.
plt.xlabel("Importance")

# Automatically scales margins, text, and labels to prevent visual clipping bugs.
plt.tight_layout()

# Deploys the interactive interface window to render the final chart on screen.
# WARNING: Script execution will pause at this line until you close the graph window!
plt.show()


# ==============================================================================
# DEPLOYMENT AND REAL-WORLD INFERENCE
# ==============================================================================
# Context: This is the ultimate goal of our machine learning pipeline. 
# We simulate a real-world app user entering fresh data for a single district.
# ==============================================================================

# --- CAPTURE NEW RAW TELEMETRY DATA ---
# We bundle the new district attributes inside a Python dictionary, then convert 
# it into a Pandas DataFrame. 
# CRITICAL FORMAT NOTE: The structure, key spelling, and column order must strictly 
# match the original 'X_train' framework, or the model will throw a shape error.
new_district = pd.DataFrame([{
    "MedInc": 5.0,       # Median income scaled (5.0 represents $50,000)
    "HouseAge": 20.0,    # Average age of structures in the neighborhood
    "AveRooms": 6.0,     # Mean number of rooms per household units
    "AveBedrms": 1.0,    # Mean number of bedrooms per household units
    "Population": 1200.0,# Total residents living within the district boundaries
    "AveOccup": 3.0,     # Average number of household members per home
    "Latitude": 34.05,   # Geographic coordinates: Los Angeles area latitude
    "Longitude": -118.25,# Geographic coordinates: Los Angeles area longitude
}])


# --- GENERATE THE REAL-TIME PREDICTION ---
# 1. final_model.predict(new_district) -> Feeds the dataframe into the 100 trees, 
#    which outputs an array containing a single predicted price value.
# 2. '[0]' -> Extracts that single floating-point number directly out of the array 
#    container so we can manipulate it mathematically.
prediction = final_model.predict(new_district)[0]


# --- RE-SCALE AND DISPLAY FOR HUMAN READABILITY ---
# The original dataset target answers were scaled down (e.g., 2.5 represented $250,000).
# We reverse this translation to display an understandable currency figure:
#   1. '* 100000' -> Converts the model's raw numerical score back into real dollars.
#   2. ':,.0f'    -> Adds commas for thousands separation and drops decimal trailing zeros.
print(f"\nPredicted median house value: ${prediction * 100000:,.0f}")
