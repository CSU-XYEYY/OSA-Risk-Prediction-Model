def save_results_to_excel(results_df, filename='RF_Regression_Prediction_Results.xlsx'):
    results_df.to_excel(filename, index=False)

def visualize_results(results_df):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Index'], results_df['Predicted'], color='blue', label='Predicted')
    plt.scatter(results_df['Index'], results_df['Actual'], color='red', label='Actual')
    plt.xlabel('Index')
    plt.ylabel('Target Value')
    plt.title('Predicted vs Actual Values (Regression)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()