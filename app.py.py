#In[1]:
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
loan_df = pd.read_csv(r"C:\Users\hoang\Downloads\dataset - loan application..csv")


#In[2]:
#streamlit run c:\Users\hoang\Downloads\app.py.py
st.title('Loan Application')
st.caption('In the context of personal finance, bank loans are a common way to fund home purchases, automobile purchases, and investments in personal initiatives. However, the loan application procedure is not always straightforward. Banks and credit organizations frequently use criteria to assess repayment capabilities and decide whether a loan is authorized or not. In this setting, the application of predictive tools to assist lending choices is critical.')
st.divider()
st.header('Example of Dataset we use to predict')
st.dataframe(loan_df.head())

if __name__ == '__main__':
    st.sidebar.text('This is the sidebar.')


#In[3]:
st.title('Dữ liệu từng giá trị')
sns.set(rc={'figure.figsize': (11.7, 8.27)})

# Create a large figure to accommodate all subplots
fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.27))

# Create subplots and display countplots using Seaborn
sns.countplot(x="Gender", hue='Loan_Status', data=loan_df, ax=axes[0, 0])
sns.countplot(x="Married", hue='Loan_Status', data=loan_df, ax=axes[0, 1])
sns.countplot(x="Education", hue='Loan_Status', data=loan_df, ax=axes[0, 2])
sns.countplot(x="Self_Employed", hue='Loan_Status', data=loan_df, ax=axes[1, 0])
sns.countplot(x="Dependents", hue='Loan_Status', data=loan_df, ax=axes[1, 1])
sns.countplot(x="Property_Area", hue='Loan_Status', data=loan_df, ax=axes[1, 2])

# You can adjust titles or labels for each subplot here (optional)

# Display the entire figure in Streamlit
st.pyplot(fig)

#In[4]:
st.title('Loan Data Visualization')

# Define bins for histograms
bins_income = np.linspace(loan_df['ApplicantIncome'].min(), loan_df['ApplicantIncome'].max(), 12)
bins_loan_term = np.linspace(loan_df['Loan_Amount_Term'].min(), loan_df['Loan_Amount_Term'].max(), 12)
bins_coapplicant_income = np.linspace(loan_df['CoapplicantIncome'].min(), loan_df['CoapplicantIncome'].max(), 12)

# Create FacetGrids and plot histograms
for feature, bins, title in [("ApplicantIncome", bins_income, "Applicant Income"),
                              ("Loan_Amount_Term", bins_loan_term, "Loan Amount Term"),
                              ("CoapplicantIncome", bins_coapplicant_income, "Coapplicant Income")]:
    st.subheader(f"Countplot for {title}")
    graph = sns.FacetGrid(loan_df, col="Gender", hue="Loan_Status", palette="Set2", col_wrap=2)
    graph.map(plt.hist, feature, bins=bins, ec="k")
    graph.axes[-1].legend()
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

#In[5]:
numeric_data = loan_df.select_dtypes(include=['number'])

# Calculate correlation matrix
correlation_matrix = numeric_data.corr()

# Create a mask for the upper triangle
mask = np.zeros_like(correlation_matrix)
mask[np.triu_indices_from(mask)] = True

# Set up Streamlit app
st.title('Correlation Matrix')

# Plot the correlation matrix
with sns.axes_style("white"):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap="YlOrRd", ax=ax)
    st.pyplot(fig)

#In[6]:
categorical_columns = ['Gender', 'Married', 'Dependents',
                       'Education', 'Self_Employed', 'Property_Area',
                       'Credit_History', 'Loan_Amount_Term']
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'Total_Income']

# Convert relevant columns to numeric
loan_df['ApplicantIncome'] = pd.to_numeric(loan_df['ApplicantIncome'], errors='coerce')
loan_df['CoapplicantIncome'] = pd.to_numeric(loan_df['CoapplicantIncome'], errors='coerce')

loan_df['Total_Income'] = loan_df['ApplicantIncome'] + loan_df['CoapplicantIncome']

# Set up Streamlit app
st.title('Boxplots for Numerical Features')

# Plot boxplots for numerical features
for feature in numerical_columns:
    # Create subplots for 'Before' and 'After'
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Before log-transformation
    axes[0].boxplot(loan_df[feature])
    axes[0].set_title(f'Before Log-Transformation: {feature}')
    axes[0].set_xlabel('Dataset')
    axes[0].set_ylabel(feature)

    # After log-transformation
    axes[1].boxplot(np.log1p(loan_df[feature]))
    axes[1].set_title(f'After Log-Transformation: {feature}')
    axes[1].set_xlabel('Dataset')
    axes[1].set_ylabel(f'Log({feature} + 1)')

    # Tight layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    st.pyplot(fig)

#In[7]
# Select numerical columns
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'Total_Income']

# Copy DataFrame for transformations
loan_transformed = loan_df.copy()

# Set up Streamlit app
st.title('Line Plots for Numerical Features')

# Plot line plots for numerical features
for feature in numerical_columns:
    # Create a new figure for each feature
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Before log-transformation
    axes[0].plot(range(len(loan_df[feature])), loan_df[feature].values)
    axes[0].set_title(f'Before Log-Transformation: {feature}')
    axes[0].set_xlabel('Dataset')
    axes[0].set_ylabel(feature)

    # After log-transformation
    loan_transformed['Total_Income'] = np.log1p(loan_transformed['Total_Income'])
    axes[1].plot(range(len(loan_transformed['Total_Income'])), loan_transformed['Total_Income'].values)
    axes[1].set_title(f'After Log-Transformation: {feature}')
    axes[1].set_xlabel('Dataset')
    axes[1].set_ylabel(f'Log({feature} + 1)')

    # Tight layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    st.pyplot(fig)
