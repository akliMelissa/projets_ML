CREDIT‑SCORING 

This project trains a machine‑learning model to predict the probability that a loan applicant will default
 and exposes the model through a simple Streamlit web app.

1. GOAL : 
   • Use the public "Give Me Some Credit" dataset 
   • Train and compare Logistic Regression, Random Forest and XGBoost models
   • Keep the model with the best ROC‑AUC score and store it 
   • Provide a friendly interface where you type applicant details and instantly see:
        – The predicted probability by default
        – A clear "HIGH RISK" or "LOW RISK" label

2. PROJECT STRUCTURE
   data : data CSV files "GiveMeSomeCredit‑training.csv"
   src : source python code
    - data_preprocessing.py : cleaning, encoding and scaling
    - model_training.py : model training , testing , selection and saving

   notebooks : contains a notebook version  "credit_scoring.ipynb" 
   app : contains a streamlit user interface "streamlit_app.py"     
   models : generated after training for saving : 
        - best_model.pkl : the best model (with the best prediction performance)
        - feature_names.pkl : columns order used by the model


3. HOW TO RUN IT

   # 1. install dependencies
   pip install -r  "requirements bellow"

   # 2. train the model : execute data_preprocessing then model_training src codes
    - python src/data_preprocessing.py
    - python src/model_training.py

   # 3. launch the web app  :  streamlit run app/streamlit_app.py
      "Note : The app will open in your browser "


4. REQUIREMENTS (with Python 3.9+)
   pandas
   scikit‑learn
   xgboost
   joblib
   streamlit


5. RESULT
   On the default train and validation split the XGBoost pipeline typically reaches best accuracy.
   The model is written to models/best_model.pkl and is automatically loaded by the Streamlit front‑end.

