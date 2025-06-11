import lime
from lime.lime_text import LimeTextExplainer

def explain_prediction(model, text):
    explainer = LimeTextExplainer(class_names=['Rejected', 'Accepted'])
    exp = explainer.explain_instance(text, model.predict_proba)
    return exp.show_in_notebook()

# SHAP integration
import shap
shap.initjs()

def shap_explanation(model, X_train, sample):
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    shap_values = explainer.shap_values(sample)
    return shap.force_plot(explainer.expected_value[0], shap_values[0], sample)
