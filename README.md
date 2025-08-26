# web_app_regression/web_app_regression/README.md

# Web App Regression

This project is a web application built using Flask that provides regression inference capabilities. It allows users to upload data, perform predictions using a pre-trained regression model, and view the results in a user-friendly interface.

## Features

- Upload CSV files for regression analysis.
- Predict target values using a trained regression model.
- Display predicted values alongside actual values.
- Simple and intuitive web interface.

## Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd web_app_regression
   ```

2. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:

   ```
   python app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Use the provided form to upload your CSV file and submit it for prediction.

4. View the results on the results page.

## Directory Structure

```
web_app_regression
├── app.py                  # Entry point for the Flask application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Files and directories to ignore in version control
├── instance
│   └── config.py          # Configuration settings for the application
├── models
│   ├── model_fold_5.pkl    # Trained regression model
│   └── poly_scaler.pkl     # Polynomial features and scaler
├── src
│   ├── RF_inference_Regression.py  # Main inference logic
│   ├── preprocessing.py           # Data preprocessing functions
│   └── utils.py                   # Utility functions
├── templates
│   ├── index.html                # Main page with input form
│   └── results.html              # Results page displaying predictions
├── static
│   ├── css
│   │   └── style.css             # Styles for the application
│   ├── js
│   │   └── app.js                # JavaScript for front-end interactions
│   └── uploads                   # Directory for uploaded files
└── tests
    ├── test_inference.py         # Unit tests for inference logic
    └── test_utils.py             # Unit tests for utility functions
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.