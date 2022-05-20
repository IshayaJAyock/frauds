#from optparse import Values
import pickle
import pandas as pd
import numpy as np
import streamlit

model_name = 'xgboost.pkl'



# load the model from disk
xg_model = pickle.load(open(model_name, 'rb'))
#xg_model = loaded_model.load(xgboost)




# load the model
# xgboost = open(model_name, 'rb')
# xg_model = joblib.load(xgboost)

# print(xg_model)


def make_prediction(data):
    
    # v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    # v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,
    # v22,v23,v24,v25,v26,v27,v28,v29,v30)

    # values = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
    # v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,
    # v22,v23,v24,v25,v26,v27,v28,v29,v30]
    pred_arr = np.array(data)
    preds = pred_arr.reshape(1, -1)
    preds = preds.astype(float)
    model_prediction = xg_model.predict(preds)

    return model_prediction


def run():
    streamlit.title("Credit Card Fraud Detection Model")
    html_temp = """
    Welcome to our crediti card fraud detection model.
    """
    streamlit.markdown(html_temp)
    var_1 = streamlit.text_input("Time")
    var_2 = streamlit.text_input("Variable 2")
    var_3 = streamlit.text_input("Variable 3")
    var_4 = streamlit.text_input("Variable 4")
    var_5 = streamlit.text_input("Variable 5")
    var_6 = streamlit.text_input("Variable 6")
    var_7 = streamlit.text_input("Variable 7")
    var_8 = streamlit.text_input("Variable 8")
    var_9 = streamlit.text_input("Variable 9")
    var_10 = streamlit.text_input("Variable 10")
    var_11 = streamlit.text_input("Variable 11")
    var_12= streamlit.text_input("Variable 12")
    var_13 = streamlit.text_input("Variable 13")
    var_14 = streamlit.text_input("Variable 14")
    var_15 = streamlit.text_input("Variable 15")
    var_16 = streamlit.text_input("Variable 16")
    var_17 = streamlit.text_input("Variable 17")
    var_18 = streamlit.text_input("Variable 18")
    var_19 = streamlit.text_input("Variable 19")
    var_20 = streamlit.text_input("Variable 20")
    var_21 = streamlit.text_input("Variable 21")
    var_22 = streamlit.text_input("Variable 22")
    var_23 = streamlit.text_input("Variable 23")
    var_24 = streamlit.text_input("Variable 24")
    var_25 = streamlit.text_input("Variable 25")
    var_26 = streamlit.text_input("Variable 26")
    var_27= streamlit.text_input("Variable 27")
    var_28 = streamlit.text_input("Variable 28")
    var_29 = streamlit.text_input("Variable 29")
    var_30 = streamlit.text_input("Amount")
    data = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, 
         var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20,
          var_21, var_22, var_23, var_24, var_25, var_26, var_27, var_28, var_29, var_30]
    print(data)
    prediction = ""
    if streamlit.button("Predict"):
        prediction = make_prediction(data)
    streamlit.success("The prediction by Model  : {}".format(prediction))

if __name__=='__main__':
    run()
