import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/tabular_model.pkl")

# Load background data
X_train = pd.read_csv("data/X_train.csv")

explainer = shap.TreeExplainer(model)

def generate_shap_plot(user_input):
    
    shap_values = explainer.shap_values(user_input)[1]

    fig, ax = plt.subplots()
    
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value[1],
            data=user_input.iloc[0],
            feature_names=user_input.columns
        )
    )
    
    return fig