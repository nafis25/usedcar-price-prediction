import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

df_preview = pd.read_csv("cars_dataset.csv")
df = pd.read_csv("cars_dataset.csv")

df.drop(df[df['mpg'] < 15].index, inplace = True)
df.drop(df[df['engineSize'] < 1].index, inplace = True)

df['fuelType'] = df['fuelType'].replace('Electric', 'Other')
df['transmission'] = df['transmission'].replace('Other', 'Manual')

inference_data = []

st.image('./pictures/cars.jpg')

st.title('How much is your rusty old car actually worth?')
st.subheader('Well, to answer that question we looked at the these features found on the UK Used Cars dataset.')
st.write('This dataset is the stacked version of [100,000 UK Used Car Data](https://www.kaggle.com/aishwaryamuthukumar/cars-dataset-audi-bmw-ford-hyundai-skoda-vw) present in Kaggle. Here we have combined the used car information of 7 brands namely Audi, BMW, Skoda, Ford, Volkswagen, Toyota and Hyundai.')
st.markdown("* **make:** the car's make")
st.markdown("* **model:** the car's model")
st.markdown('* **transmission:** the transmission mode of the car')
st.markdown('* **mileage:** the number of miles the car has been driven')
st.markdown('* **fuelType:** fuel type of the car')
st.markdown('* **tax:** annual tax to be paid for the car')
st.markdown('* **mpg:** miles per the gallon the car can travel')
st.markdown('* **year:** the year the car was manufactured in')
st.markdown('* **engineSize:** engine cylinder capacity of the car')

st.subheader('Target')
st.markdown('* **price:** price of the listed car')

st.subheader("Here's what the data looks like in a DataFrame")
st.write(df_preview.head(10))

st.header("Begin by filling in your car's specs:")

st.subheader("What's your car's manufacturing year?")
car_year = st.selectbox("", sorted(df.year.unique()))

#Dictionary to map out the models for a specific make
make_model_map = {}

for make in df.Make.unique():
  model_list = df[df['Make']==make]['model'].unique()
  make_model_map[make] = list(model_list)

st.subheader("What's your car's engine make?")
car_make = st.selectbox("", sorted(df.Make.unique()))


st.subheader("What's your car's engine model?")
car_model = st.selectbox("", sorted(make_model_map[car_make]))


st.subheader("What's your car's engine transmission?")
car_transmission = st.radio("", df.transmission.unique())


st.subheader("What's your car's fuel type?")
car_fueltype = st.radio("", df.fuelType.unique())

st.subheader("What's your car's engine mileage?")
car_mileage = st.slider("", min_value = df.mileage.min(), max_value = df.mileage.max(), value = 50000)

st.subheader("How much road tax do you pay yearly for your car?")
car_tax = st.slider("", min_value = df.tax.min(), max_value = df.tax.max(), value = 400.00)


st.subheader("How many miles per gallon does your car go?")
car_mpg = st.slider("", min_value = df.mpg.min(), max_value = df.mpg.max(), value = 130.00)


st.subheader("What's your car's engine size?")
car_enginesize = st.slider("",min_value = df.engineSize.min(), max_value = df.engineSize.max(), value = 4.50)


st.subheader("Select the Algorithm you want to infer with")
selected_model = st.selectbox("",["Decision Tree Regressor", "Linear Regression", "ElasticNet", "Ridge"])


#Preprocess Data
inference_data.append([car_model,car_year,car_transmission,car_mileage,car_fueltype,car_tax,car_mpg,car_enginesize,car_make])
inference_data_df = pd.DataFrame(inference_data,columns=['model','year','transmission','mileage','fuelType','tax','mpg','engineSize','Make'])
cat_cols = list(df.select_dtypes('object').columns)
num_cols = [cols for cols in df.columns if cols not in cat_cols]
le =LabelEncoder()
for c in cat_cols:
  inference_data_df[c] = le.fit_transform(inference_data_df[c])

#Function to generate predictions
def model_switcher(model):

    if model == "Decision Tree Regressor":
        loaded_model = joblib.load("models/dtr_model.pkl")
        result = loaded_model.predict(inference_data_df[0:1])
        return "£" + str(round(abs(result[0]),2))

    elif model == "Linear Regression":
        loaded_model = joblib.load("models/lr_model.pkl")
        result = loaded_model.predict(inference_data_df[0:1])
        return "£" + str(round(abs(result[0]),2))

    elif model == "ElasticNet":
        loaded_model = joblib.load("models/Elastic_model.pkl")
        result = loaded_model.predict(inference_data_df[0:1])
        return "£" + str(round(abs(result[0]),2))

    elif model == "Ridge":
        loaded_model = joblib.load("models/Ridge_model.pkl")
        result = loaded_model.predict(inference_data_df[0:1])
        return "£" + str(round(abs(result[0]),2))

    else:
        return "Baal"

st.subheader("Your car could be worth....")
st.markdown("# _{}_".format(model_switcher(selected_model)))





