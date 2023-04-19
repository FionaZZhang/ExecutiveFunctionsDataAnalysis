import csv
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cold_path = 'Track3Data/CognitiveData_Cold.csv'
hot_path = 'Track3Data/CognitiveData_Hot.csv'
inspect_output_path_cold = 'Cold_Data_Inspect.txt'
inspect_output_path_hot = 'Hot_Data_Inspect.txt'

#======================================================================================================================
# Data inspection
with open(cold_path) as csvfile, open(inspect_output_path_cold, 'w') as outfile:
    reader = csv.reader(csvfile)
    header = next(reader) # read the header row
    data = list(reader)   # read the remaining rows into a list

    # Write column names to file
    outfile.write("Column names: " + ", ".join(header) + "\n\n")

    # Calculate statistics for each column
    for i, col_name in enumerate(header):
        # Convert column data to float
        col_data = [float(row[i]) if row[i] != '' else np.nan for row in data]

        # Calculate statistics
        avg = np.nanmean(col_data)
        num_missing = np.isnan(col_data).sum()
        max_val = np.nanmax(col_data)
        min_val = np.nanmin(col_data)
        num_val = len(col_data)

        # Write statistics to file
        outfile.write(f"Column name: {col_name}\n")
        outfile.write(f"\tNumber of values: {num_val}\n")
        outfile.write(f"\tAverage: {avg:.2f}\n")
        outfile.write(f"\tNumber of missing values: {num_missing}\n")
        outfile.write(f"\tMaximum value: {max_val:.2f}\n")
        outfile.write(f"\tMinimum value: {min_val:.2f}\n\n")

with open(cold_path) as csvfile, open(inspect_output_path_hot, 'w') as outfile:
    reader = csv.reader(csvfile)
    header = next(reader) # read the header row
    data = list(reader)   # read the remaining rows into a list

    # Write column names to file
    outfile.write("Column names: " + ", ".join(header) + "\n\n")

    # Calculate statistics for each column
    for i, col_name in enumerate(header):
        # Convert column data to float
        col_data = [float(row[i]) if row[i] != '' else np.nan for row in data]

        # Calculate statistics
        avg = np.nanmean(col_data)
        num_missing = np.isnan(col_data).sum()
        max_val = np.nanmax(col_data)
        min_val = np.nanmin(col_data)
        num_val = len(col_data)

        # Write statistics to file
        outfile.write(f"Column name: {col_name}\n")
        outfile.write(f"\tNumber of values: {num_val}\n")
        outfile.write(f"\tAverage: {avg:.2f}\n")
        outfile.write(f"\tNumber of missing values: {num_missing}\n")
        outfile.write(f"\tMaximum value: {max_val:.2f}\n")
        outfile.write(f"\tMinimum value: {min_val:.2f}\n\n")

#======================================================================================================================
# Read in the two CSV files
df1 = pd.read_csv(cold_path)
df2 = pd.read_csv(hot_path)

# Read in the two CSV files
df1 = pd.read_csv(cold_path)
df2 = pd.read_csv(hot_path)

# Select the columns based on their column names that start with 'Shifting' and 'Memory'
selected_cols1 = [col for col in df1.columns if col.startswith('Shifting') or col.startswith('Memory')]
selected_cols2 = [col for col in df2.columns if col.startswith('Shifting') or col.startswith('Memory')]

# Standardize and normalize the selected columns in each file
scaler = StandardScaler()
min_max_scaler = MinMaxScaler()
df1_selected_cols = pd.DataFrame(min_max_scaler.fit_transform(scaler.fit_transform(df1[selected_cols1])), columns=selected_cols1)
df2_selected_cols = pd.DataFrame(min_max_scaler.transform(scaler.transform(df2[selected_cols2])), columns=selected_cols2)

# Add the 'User' column to each dataframe
df1_selected_cols.insert(0, 'User', df1['User'])
df2_selected_cols.insert(0, 'User', df2['User'])

# Add the 'stress_level' column to each dataframe
df1_selected_cols.insert(1, 'stress_level', 0)
df2_selected_cols.insert(1, 'stress_level', 1)

# Concatenate the two dataframes vertically
df_concat = pd.concat([df1_selected_cols, df2_selected_cols], ignore_index=True, axis=0)

# Write the concatenated and standardized dataframe to a new CSV file
df_concat.to_csv('Concatenated_Data.csv', index=False)

#======================================================================================================================
# Read in the concatenated CSV file
df_concat = pd.read_csv('Concatenated_Data.csv')

# Select the columns to plot
plot_cols = [col for col in df_concat.columns if col not in ['User', 'stress_level']]

# Calculate the average values for stress_level = 0 and stress_level = 1
mean_vals_0 = df_concat[df_concat['stress_level'] == 0][plot_cols].mean()
mean_vals_1 = df_concat[df_concat['stress_level'] == 1][plot_cols].mean()

# Create a bar plot with two bars for each column
x = list(range(len(plot_cols)))
bar_width = 0.35
fig, ax = plt.subplots()
ax.bar(x, mean_vals_0, width=bar_width, label='stress_level = 0')
ax.bar([i + bar_width for i in x], mean_vals_1, width=bar_width, label='stress_level = 1')
ax.set_xticks([i + bar_width / 2 for i in x])
ax.set_xticklabels(plot_cols, rotation=45)
ax.legend()
# plt.show()

# Separate the columns starting with 'Shifting' and 'Memory'
selected_cols_Shifting = [col for col in df_concat.columns if col.startswith('Shifting')]
selected_cols_Memory = [col for col in df_concat.columns if col.startswith('Memory')]

# Compute the correlation matrices
corr_Shifting = df_concat[selected_cols_Shifting + ['stress_level']].corr()
corr_Memory = df_concat[selected_cols_Memory + ['stress_level']].corr()

# Plot the heatmaps
sns.set(font_scale=1.2)
plt.figure(figsize=(12, 8))
sns.heatmap(corr_Shifting, cmap='coolwarm', center=0, annot=False, cbar_kws={'label': 'Correlation'})
plt.title('Correlation Heatmap (Cold)')
plt.savefig('Corr_Shifting.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(12, 8))
sns.heatmap(corr_Memory, cmap='coolwarm', center=0, annot=False, cbar_kws={'label': 'Correlation'})
plt.title('Correlation Heatmap (Hot)')
plt.savefig('Corr_Memory.png', dpi=300, bbox_inches='tight')

#=========================================================================================================
# combine columns with matching prefixes and suffixes
df_combine = df_concat
# Get the columns that we want to combine using regular expressions
shifting_congruent_rt_cols = df_combine.filter(regex=r'^Shifting.*Congruent_RT$').columns
shifting_incongruent_rt_cols = df_combine.filter(regex=r'^Shifting.*Incongruent_RT$').columns
shifting_congruent_fa_cols = df_combine.filter(regex=r'^Shifting.*Congruent_FA$').columns
shifting_incongruent_fa_cols = df_combine.filter(regex=r'^Shifting.*Incongruent_FA$').columns
# shifting_accuracy_cols = df_combine.filter(regex=r'^Shifting.*Accuracy$').columns
# shifting_errors_cols = df_combine.filter(regex=r'^Shifting.*Errors$').columns
memory_hits_cols = df_combine.filter(regex=r'^Memory.*Hits$').columns
memory_pr_cols = df_combine.filter(regex=r'^Memory.*PR$').columns
memory_attempts_cols = df_combine.filter(regex=r'^Memory.*Attempts$').columns
memory_fa_cols = df_combine.filter(regex=r'^Memory.*FA$').columns
memory_efficiency = df_combine.filter(regex=r'^Memory.*Return_Efficiency$').columns


# Combine the columns using the mean
df_combine['Shifting_Congruent_RT'] = df_combine[shifting_congruent_rt_cols].mean(axis=1)
df_combine['Shifting_Incongruent_RT'] = df_combine[shifting_incongruent_rt_cols].mean(axis=1)
df_combine['Shifting_Congruent_FA'] = df_combine[shifting_congruent_fa_cols].mean(axis=1)
df_combine['Shifting_Incongruent_FA'] = df_combine[shifting_incongruent_fa_cols].mean(axis=1)
#df_combine['Shifting_Accuracy'] = df_combine[shifting_accuracy_cols].mean(axis=1)
#df_combine['Shifting_Errors'] = df_combine[shifting_errors_cols].mean(axis=1)
df_combine['Memory_Hits'] = df_combine[memory_hits_cols].mean(axis=1)
df_combine['Memory_PR'] = df_combine[memory_pr_cols].mean(axis=1)
df_combine['Memory_Attempts'] = df_combine[memory_attempts_cols].mean(axis=1)
df_combine['Memory_FA'] = df_combine[memory_fa_cols].mean(axis=1)
df_combine['Memory_Return_Efficiency'] = df_combine[memory_efficiency].mean(axis=1)


# Select the columns we want to keep in the new CSV
new_cols = ['User', 'stress_level', 'Shifting_Congruent_RT', 'Shifting_Incongruent_RT', 'Shifting_Congruent_FA',
            'Shifting_Incongruent_FA',
            'Memory_Hits', 'Memory_PR', 'Memory_Attempts', 'Memory_FA', 'Memory_Return_Efficiency']
df_new_comb = df_combine[new_cols]

# Write the updated CSV to a file
df_new_comb.to_csv('Combined_Data.csv', index=False)

# Calculate the correlation coefficients
corr_matrix = df_new_comb.corr()

# Filter the correlation matrix to only show correlations with stress_level
corr_with_stress = corr_matrix.loc['stress_level']

# Print the correlations between stress_level and the other columns
print(corr_with_stress)

fig, ax = plt.subplots(3, 3, figsize=(20, 20))
for row in range(3):
    for col in range(3):
        i = 3 * row + col + 2
        ax[row, col].scatter(df_new_comb['stress_level'], df_new_comb.iloc[:, i])
        ax[row, col].set_xlabel('stress_level')
        ax[row, col].set_ylabel(df_new_comb.columns[i])

plt.tight_layout()
plt.savefig('Scatter_stress_shifting_memory')
# plt.show()

#=========================================================================================================
#Get difference

import pandas as pd

# Read the input CSV file
input_csv = 'Combined_Data.csv'
output_csv = 'Diff_Data.csv'

df = pd.read_csv(input_csv)

# Group by user and filter users that appear twice
user_counts = df['User'].value_counts()
users_to_process = user_counts[user_counts == 2].index

# Initialize an empty dataframe to store the results
result_df = pd.DataFrame(
    columns=['User', 'Shifting_Congruent_RT_Diff', 'Shifting_Incongruent_RT_Diff', 'Shifting_Congruent_FA_Diff',
             'Shifting_Incongruent_FA_Diff', 'Memory_Hits_Diff', 'Memory_PR_Diff', 'Memory_Attempts_Diff',
             'Memory_FA_Diff', 'Memory_Return_Efficiency_Diff'])

# Process each user
for user in users_to_process:
    user_data = df[df['User'] == user]
    stress_0 = user_data[user_data['stress_level'] == 0].reset_index(drop=True)
    stress_1 = user_data[user_data['stress_level'] == 1].reset_index(drop=True)

    # Calculate the differences
    diff_data = stress_0.loc[:, 'Shifting_Congruent_RT':'Memory_Return_Efficiency'] - stress_1.loc[:,
                                                                                      'Shifting_Congruent_RT':'Memory_Return_Efficiency']

    # Add user ID and rename the columns
    diff_data['User'] = user
    diff_data.columns = [col + '_Diff' if col != 'User' else col for col in diff_data.columns]

    # Append the result to the result dataframe
    result_df = result_df.append(diff_data, ignore_index=True)

# Save the results to a new CSV file
result_df.to_csv(output_csv, index=False)

# Create a pair plot to visualize the relationships between the columns
sns.pairplot(result_df, diag_kind='hist', corner=True)

# Show the plot
plt.savefig('Diff_Pairplot.png')

# =========================================================================================================
# Getting age and gender data
import pandas as pd

# Read in the first CSV file
df1 = pd.read_csv('Track3Data/Baseline_selfreport_age modified(1) 2.csv', encoding='ISO-8859-1')

# Read in the second CSV file
df2 = pd.read_csv('Track3Data/SID participant name.csv', encoding='ISO-8859-1')

# Extract first names from df1
df1['Name'] = df1['Q2']
df1['Age'] = df1['Q3']
df1 = df1[['Name', 'Age']]
df1 = df1[df1.index % 2 == 1]

# Extract first names from df2
df2['Name'] = df2['Participant (First Last)']
df2 = df2[['Subject ID', 'Name', 'Gender']]

# Merge the data based on first name
merged_df = pd.merge(df1, df2, on='Name', how='inner')

# Print the number of rows in each dataframe before and after the merge
print("Number of rows in df1:", len(df1))
print("Number of rows in df2:", len(df2))
print("Number of rows in merged_df:", len(merged_df))

# Select only the desired columns
final_df = merged_df[['Subject ID', 'Name', 'Age', 'Gender']]

# Write out the final CSV file
final_df.to_csv('Age_Gender.csv', index=False)

# Combining data into one csv
# Read in the two CSV files
file1 = pd.read_csv('Diff_Data.csv')
file2 = final_df

# Merge the two dataframes
merged = pd.merge(file2, file1, left_on='Subject ID', right_on='User', how='inner')
merged = merged.drop(columns=['Subject ID'])

# Write out the merged dataframe to a new CSV file
merged['Age'] = merged['Age'].astype(float)
merged.to_csv('Preped_Data.csv', index=False)


# =========================================================================================================
# Linear Regression
import statsmodels.api as sm
from tabulate import tabulate

# Read in the CSV file
df = merged

# Select only the columns of interest
cols = ['Age', 'Shifting_Congruent_RT_Diff', 'Shifting_Incongruent_RT_Diff', 'Shifting_Congruent_FA_Diff', 'Shifting_Incongruent_FA_Diff', 'Memory_Hits_Diff', 'Memory_PR_Diff', 'Memory_Attempts_Diff', 'Memory_FA_Diff', 'Memory_Return_Efficiency_Diff']
df = df[cols]

# Create a new text file for the performance summary
with open('performance_summary_age.txt', 'w') as f:
    # Write the header row
    f.write('{:<30} {:<15} {:<15} {:<15} {:<15} {:<15}\n'.format('Column', 'R-squared', 'Adj. R-squared', 'F-statistic',
                                                                 'Prob (F-statistic)', 'Coefficients'))

    # Perform linear regression for each column
    for col in cols[1:]:
        X = df[['Age']]
        y = df[col]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        # Get the relevant values
        rsquared = round(model.rsquared, 3)
        adj_rsquared = round(model.rsquared_adj, 3)
        f_stat = round(model.fvalue, 3)
        p_value = round(model.f_pvalue, 3)
        coef = round(model.params[1], 3)

        # Write the row for the current column
        f.write(
            '{:<30} {:<15} {:<15} {:<15} {:<15} {:<15}\n'.format(col, rsquared, adj_rsquared, f_stat, p_value, coef))

# Create a list of plot types to generate
plot_types = ['scatter', 'regplot', 'residplot']

# Iterate over the plot types
for plot_type in plot_types:
    # Create a figure with subplots for each column
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    axes = axes.flatten()

    # Iterate over the columns
    for i, col in enumerate(cols[1:]):
        # Select the appropriate subplot
        ax = axes[i]

        # Create the plot
        if plot_type == 'scatter':
            ax.scatter(df['Age'], df[col])
            ax.set_xlabel('Age')
            ax.set_ylabel(col)
        elif plot_type == 'regplot':
            sns.regplot(x='Age', y=col, data=df, ax=ax)
        elif plot_type == 'residplot':
            sns.residplot(x='Age', y=col, data=df, ax=ax)

        # Add a title to the subplot
        ax.set_title(f'{plot_type.capitalize()} of {col}')

    # Adjust the layout of the subplots and save the figure
    plt.tight_layout()
    plt.savefig(f'{plot_type}_plots.png')

# =========================================================================================================
# ANOVA analysis
from scipy.stats import f_oneway

# Read in the CSV file
df = merged

# Select only the columns of interest
cols = ['Gender', 'Shifting_Congruent_RT_Diff', 'Shifting_Incongruent_RT_Diff', 'Shifting_Congruent_FA_Diff', 'Shifting_Incongruent_FA_Diff', 'Memory_Hits_Diff', 'Memory_PR_Diff', 'Memory_Attempts_Diff', 'Memory_FA_Diff', 'Memory_Return_Efficiency_Diff']
df = df[cols]

# Group the data by gender
grouped = df[cols].groupby('Gender')

# Perform the ANOVA analysis on each column
results = {}
for col in cols[1:]:
    col_data = [grouped.get_group('Male')[col], grouped.get_group('Female')[col]]
    f_stat, p_val = f_oneway(*col_data)
    results[col] = {'F-statistic': f_stat, 'p-value': p_val}

# Save the ANOVA results to a text file
with open('anova_result.txt', 'w') as f:
    for col, result in results.items():
        f.write(f"{col}:\n")
        f.write(f"\tF-statistic: {result['F-statistic']:.3f}\n")
        f.write(f"\tp-value: {result['p-value']:.3f}\n\n")


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
axes = axes.flatten()

for i, col in enumerate(cols[1:]):
    sns.boxplot(x='Gender', y=col, data=df, ax=axes[i])
    axes[i].set_xlabel('Gender')
    axes[i].set_ylabel(col)
    axes[i].set_title(f'{col}')

plt.tight_layout()
plt.savefig(f'gender_boxplots.png')

# =========================================================================================================
# Correlation


# =========================================================================================================
# Neural Network
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

# Load the data
data = pd.read_csv('Preped_Data.csv')

# Separate the input features (X) and the output (y)
X = data.iloc[:, [1, 2]].values
Y = data.iloc[:, -2].values

male_indices = np.where(X[:, 1] == 'Male')[0]
female_indices = np.where(X[:, 1] == 'Female')[0]
num_males = len(male_indices)
num_females = len(female_indices)
print(f"Number of males: {num_males}")
print(f"Number of females: {num_females}")

# One hot encode the Gender column in X
gender_encoder = OneHotEncoder()
gender_encoded = gender_encoder.fit_transform(X[:, 1].reshape(-1, 1)).toarray()
X = np.concatenate((X[:, [0]], gender_encoded), axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()



# Define the model architecture
class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


model = MyModel(X_train_tensor.shape[1])

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_losses = []
val_losses = []
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor.unsqueeze(1))
    loss.backward()
    optimizer.step()

    train_loss = loss.item()
    val_loss = criterion(model(X_test_tensor), y_test_tensor.unsqueeze(1)).item()
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss,
                                                                                     val_loss))

# Evaluate the model
with torch.no_grad():
    predicted = model(X_test_tensor)
    mse = criterion(predicted, y_test_tensor.unsqueeze(1))
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(predicted - y_test_tensor.unsqueeze(1)))
    print('Mean Squared Error: {:.4f}'.format(mse.item()))
    print('Root Mean Squared Error: {:.4f}'.format(rmse.item()))
    print('Mean Absolute Error: {:.4f}'.format(mae.item()))


# Plot the loss curves
plt.clf()
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curves.png')

# Plot the prediction vs. actual values
plt.clf()
plt.scatter(y_test, predicted.numpy(), alpha=0.5)
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Prediction vs. Actual')
plt.savefig('prediction_actual.png')

# # Calculate permutation feature importance
# result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
# importance = result.importances_mean
#
# # Plot the feature importance
# plt.clf()
# plt.bar(range(X.shape[1]), importance)
# plt.xticks(range(X.shape[1]), data.columns[:-1], rotation=90)
# plt.xlabel('Feature')
# plt.ylabel('Importance')
# plt.title('Feature Importance')
# plt.savefig('feature_importance.png')

# # Visualize the model architecture using TensorBoard
# from torch.utils.tensorboard import SummaryWriter
#
# writer = SummaryWriter()
# writer.add_graph(model, X_test_tensor)
# writer.close()


