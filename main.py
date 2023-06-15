from fastapi import FastAPI
import tensorflow as tf
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

app = FastAPI()
cred = credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://datausers.firebaseio.com'
})

#import model h5 disini, sebagai contoh inputan model linear
model = tf.keras.models.load_model('./recommender_model.h5')
#Import the data set
#df = pd.read_csv('./Reviews3.csv')
#read data
datas = db.reference('/dataa').get()
df = pd.DataFrame.from_dict(datas)

#buat basemodel untuk post
class DataInput(BaseModel):
    data: str

#import model h5 disini, sebagai contoh inputan model linear
model = tf.keras.models.load_model('./recommender_model.h5')

@app.get("/")
def hello():
    return {"message": "MODEL API"}

#contoh untuk get model predict
@app.get("/predict")
def predict():
    data = 'A3SGXH7AUHU8GW'
    if data in df['UserId'] and df[df['UserId'] == data].shape[0] < 5:
        # Top 10 based on rating
        most_rated = df.groupby('ProductId').size().sort_values(ascending=False)[:10]
        final_result = most_rated.index.tolist()
    else:
        # Create a dictionary mapping user IDs to unique indices
        user_ids = df['UserId'].unique()
        user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
        index_to_user_id = {index: user_id for index, user_id in enumerate(user_ids)}

        # Create a dictionary mapping product IDs to unique indices
        product_ids = df['ProductId'].unique()
        product_id_to_index = {product_id: index for index, product_id in enumerate(product_ids)}
        index_to_product_id = {index: product_id for index, product_id in enumerate(product_ids)}
        
        #----------------add dictionary mapping product id to new data
        # Create the dictionary mapping
        product_Cals_per100grams = dict(zip(df['ProductId'], df['Cals_per100grams']))
        # Create the dictionary mapping
        product_FoodCategory = dict(zip(df['ProductId'], df['FoodCategory']))
        # Create the dictionary mapping
        product_FoodName = dict(zip(df['ProductId'], df['FoodName']))
        # Create the dictionary mapping
        product_Image = dict(zip(df['ProductId'], df['Image']))
        # Create the dictionary mapping
        product_KJ_per100grams = dict(zip(df['ProductId'], df['KJ_per100grams']))
        # Create the dictionary mapping
        product_LinkToko = dict(zip(df['ProductId'], df['LinkToko']))
        #-----------------add dictionary mapping product id to new data

        # Convert user and product IDs to indices and food names in the dataframe
        df['user_index'] = df['UserId'].map(user_id_to_index)
        df['product_index'] = df['ProductId'].map(product_id_to_index)

        uidx= df['user_index'].values.astype(np.int64)
        pidx = df['product_index'].values.astype(np.int64)

        # Create a new DataFrame with converted data arrays
        df_converted = pd.DataFrame({'UserId': uidx, 'ProductId': pidx, 'Score': 0})

        # Create pivot table with the converted DataFrame
        final_ratings_matrix = pd.pivot_table(df_converted, index='UserId', columns='ProductId', values='Score')
        final_ratings_matrix.fillna(0, inplace=True)

        array3 = final_ratings_matrix.reset_index().melt(id_vars=['UserId'], value_vars=final_ratings_matrix.columns)
        array3 = array3[['UserId', 'ProductId']].values.astype(np.int64)

        # Filter the array3 for the specific user ID
        filtered_array3 = array3[array3[:, 0] == user_id_to_index[data]]

        # Perform predictions
        predictions = model.predict(filtered_array3)

        # Inverse transform the scaled ratings to get the actual ratings
        scaler = MinMaxScaler()
        score = scaler.fit_transform(df['Score'].values.reshape(-1, 1))
        predictions = scaler.inverse_transform(predictions)

        # Make prediction result to df
        df_predicted = pd.DataFrame(filtered_array3, columns=['UserId', 'ProductId'])
        df_predicted['PredictedRatings'] = predictions
        df_predicted = df_predicted.sort_values(by='PredictedRatings', ascending=False)

        # Rename the columns back to 'UserId' and 'ProductId'
        df_predicted = df_predicted.rename(columns={'UserId': 'user_index', 'ProductId': 'product_index'})

        # Convert the user index back to 'UserId' and product index back to 'ProductId'
        df_predicted['UserId'] = df_predicted['user_index'].map(index_to_user_id)
        df_predicted['ProductId'] = df_predicted['product_index'].map(index_to_product_id)

        #-----------------add new column
        df_predicted['Cals_per100grams'] = df_predicted['ProductId'].map(product_Cals_per100grams)
        df_predicted['FoodCategory'] = df_predicted['ProductId'].map(product_FoodCategory)
        df_predicted['FoodName'] = df_predicted['ProductId'].map(product_FoodName)
        df_predicted['Image'] = df_predicted['ProductId'].map(product_Image)
        df_predicted['KJ_per100grams'] = df_predicted['ProductId'].map(product_KJ_per100grams)
        df_predicted['LinkToko'] = df_predicted['ProductId'].map(product_LinkToko)
        #-----------------add new column

        final_result = df_predicted[:10].values.tolist()

    # Convert predictions to a JSON response
    response = {'predictions': final_result}
    return response


#contoh untuk post model predict
@app.post("/predict")
def predict(data: DataInput):
    data = data.data
    if data in df['UserId'] and df[df['UserId'] == data].shape[0] < 5:
        # Top 10 based on rating
        most_rated = df.groupby('ProductId').size().sort_values(ascending=False)[:10]
        final_result = most_rated.index.tolist()
    else:
        # Create a dictionary mapping user IDs to unique indices
        user_ids = df['UserId'].unique()
        user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
        index_to_user_id = {index: user_id for index, user_id in enumerate(user_ids)}

        # Create a dictionary mapping product IDs to unique indices
        product_ids = df['ProductId'].unique()
        product_id_to_index = {product_id: index for index, product_id in enumerate(product_ids)}
        index_to_product_id = {index: product_id for index, product_id in enumerate(product_ids)}
        
        #----------------add dictionary mapping product id to new data
        # Create the dictionary mapping
        product_Cals_per100grams = dict(zip(df['ProductId'], df['Cals_per100grams']))
        # Create the dictionary mapping
        product_FoodCategory = dict(zip(df['ProductId'], df['FoodCategory']))
        # Create the dictionary mapping
        product_FoodName = dict(zip(df['ProductId'], df['FoodName']))
        # Create the dictionary mapping
        product_Image = dict(zip(df['ProductId'], df['Image']))
        # Create the dictionary mapping
        product_KJ_per100grams = dict(zip(df['ProductId'], df['KJ_per100grams']))
        # Create the dictionary mapping
        product_LinkToko = dict(zip(df['ProductId'], df['LinkToko']))
        #-----------------add dictionary mapping product id to new data

        # Convert user and product IDs to indices and food names in the dataframe
        df['user_index'] = df['UserId'].map(user_id_to_index)
        df['product_index'] = df['ProductId'].map(product_id_to_index)

        uidx= df['user_index'].values.astype(np.int64)
        pidx = df['product_index'].values.astype(np.int64)

        # Create a new DataFrame with converted data arrays
        df_converted = pd.DataFrame({'UserId': uidx, 'ProductId': pidx, 'Score': 0})

        # Create pivot table with the converted DataFrame
        final_ratings_matrix = pd.pivot_table(df_converted, index='UserId', columns='ProductId', values='Score')
        final_ratings_matrix.fillna(0, inplace=True)

        array3 = final_ratings_matrix.reset_index().melt(id_vars=['UserId'], value_vars=final_ratings_matrix.columns)
        array3 = array3[['UserId', 'ProductId']].values.astype(np.int64)

        # Filter the array3 for the specific user ID
        filtered_array3 = array3[array3[:, 0] == user_id_to_index[data]]

        # Perform predictions
        predictions = model.predict(filtered_array3)

        # Inverse transform the scaled ratings to get the actual ratings
        scaler = MinMaxScaler()
        score = scaler.fit_transform(df['Score'].values.reshape(-1, 1))
        predictions = scaler.inverse_transform(predictions)

        # Make prediction result to df
        df_predicted = pd.DataFrame(filtered_array3, columns=['UserId', 'ProductId'])
        df_predicted['PredictedRatings'] = predictions
        df_predicted = df_predicted.sort_values(by='PredictedRatings', ascending=False)

        # Rename the columns back to 'UserId' and 'ProductId'
        df_predicted = df_predicted.rename(columns={'UserId': 'user_index', 'ProductId': 'product_index'})

        # Convert the user index back to 'UserId' and product index back to 'ProductId'
        df_predicted['UserId'] = df_predicted['user_index'].map(index_to_user_id)
        df_predicted['ProductId'] = df_predicted['product_index'].map(index_to_product_id)

        #-----------------add new column
        df_predicted['Cals_per100grams'] = df_predicted['ProductId'].map(product_Cals_per100grams)
        df_predicted['FoodCategory'] = df_predicted['ProductId'].map(product_FoodCategory)
        df_predicted['FoodName'] = df_predicted['ProductId'].map(product_FoodName)
        df_predicted['Image'] = df_predicted['ProductId'].map(product_Image)
        df_predicted['KJ_per100grams'] = df_predicted['ProductId'].map(product_KJ_per100grams)
        df_predicted['LinkToko'] = df_predicted['ProductId'].map(product_LinkToko)
        #-----------------add new column

        final_result = df_predicted[:10].values.tolist()

    # Convert predictions to a JSON response
    response = {'predictions': final_result}
    return response







