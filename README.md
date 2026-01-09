# XGBoost Classification Web Application

This is a Flask-based web application for online inference of XGBoost classification models. The application is built based on the structure of RF/web_app_regression, maintaining the same interface style and functional logic, but replacing the regression model with an XGBoost classification model.

## Features

- **Online Inference**: Supports real-time prediction of XGBoost classification models through a web interface
- **Multiple Input Formats**: Supports CSV format and JSON format data input
- **User-Friendly Interface**: Maintains consistent interface style with RF/web_app_regression
- **Real-time Results Display**: Prediction results are displayed with color coding (green: No or mild OSA, red: Moderate to severe OSA)

## Project Structure

```
XGboost_light/web_app_classification/
├── app.py                    # Flask main application
├── xgboost_classifier_final.pkl  # XGBoost classification model file
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html           # Frontend page template
├── static/
│   ├── css/
│   │   └── style.css        # Stylesheet
│   └── js/
│       └── app.js           # Frontend JavaScript
├── test_app.py              # Application test script
└── README.md                # Project documentation
```

## Installation and Running

### 1. Environment Requirements
- Python 3.8+
- Virtual environment recommended

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```

The application will start at http://127.0.0.1:5000.

### 4. Test the Application
```bash
python test_app.py
```

## Usage Instructions

### Input Data Format
The application supports two input formats:

1. **CSV Format** (no header):
   ```
   4.22,1.71,0.74,17.8,21,23.4,43,1
   5.5,2.0,1.2,18.5,45,25.0,45,2
   ```

2. **JSON Format**:
   ```json
   {
     "data": [
       [4.22, 1.71, 0.74, 17.8, 21, 23.4, 43, 1],
       [5.5, 2.0, 1.2, 18.5, 45, 25.0, 45, 2]
     ]
   }
   ```

### Feature Description
The model requires the following 8 features (in order):
1. Glu(mmol/L) - Blood glucose
2. FIB(mg/dL) - Fibrinogen
3. AST/ALT - Aspartate aminotransferase/Alanine aminotransferase ratio
4. AG(mmol/L) - Anion gap
5. Age - Age
6. BMI(kg/m^2) - Body Mass Index
7. NC(cm) - Neck circumference
8. Mallampati - Mallampati score

### Prediction Results
- **Class 0**: "No or mild OSA" - Displayed in green
- **Class 1**: "Moderate to severe OSA" - Displayed in red

## Technical Implementation

### Model Compatibility Handling
Due to XGBoost version compatibility issues, the application includes the following processing:
1. Automatic detection of XGBoost classifier
2. Setting necessary compatibility attributes (use_label_encoder=False, gpu_id=-1, etc.)
3. Providing alternative prediction methods to handle possible prediction errors

### Data Processing Pipeline
1. Receive and parse input data
2. Handle missing values (using median imputation)
3. Label encode categorical features
4. Rename columns to match model's expected feature names
5. Load model and make predictions
6. Return prediction results and labels

## Main Differences from RF/web_app_regression

1. **Model Type**: Changed from regression model to classification model
2. **Output Format**: Changed from continuous values to classification labels
3. **Results Display**: Changed from numerical display to color-coded label display
4. **Threshold Processing**: Uses fixed classification threshold (0.75) for label assignment

## Troubleshooting

### Common Issues
1. **Model Loading Failure**: Ensure the `xgboost_classifier_final.pkl` file exists and is readable
2. **Prediction Error**: Check if the input data has the correct number of columns and format
3. **Dependency Issues**: Ensure all dependencies are correctly installed

### Debug Mode
The application runs in debug mode and will automatically reload code changes. To disable debug mode, change `app.run(debug=True)` to `app.run(debug=False)` in `app.py`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.