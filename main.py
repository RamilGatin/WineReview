import numpy as np
import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error


from catboost import Pool, CatBoostRegressor, cv


import matplotlib.pyplot as plt
import seaborn as sns


def load_data(dataframe):
    data = pd.read_csv(dataframe)
    return data


data = load_data("winemag-data-130k-v2.csv")

st.title("Датасет")
st.dataframe(data.head())

st.markdown("""---""")

data = data.drop(columns=['Unnamed: 0', 'description'])
data = data.reset_index(drop=True)

data = data.drop_duplicates(['title'])
data = data.reset_index(drop=True)

data = data.dropna(subset=['price'])
data = data.reset_index(drop=True)

st.title("Визуализация данных")


def pastel_plot(data, x, y):
    plt.figure(figsize=(15, 6))
    plt.title('Points histogram - whole dataset')
    sns.set_color_codes("pastel")
    sns.barplot(x=x, y=y, data=df)
    locs, labels = plt.xticks()
    plt.show()


temp = data["points"].value_counts()
df = pd.DataFrame({'points': temp.index,
                   'number_of_wines': temp.values
                   })

pastel_plot(df, 'points', 'number_of_wines')

st.pyplot(plt)
plt.clf()

st.markdown("""---""")
st.title("Распределение цены")

plt.figure(figsize=(20,5))
plt.title("Distribution of price")
ax = sns.histplot(data["price"])
st.pyplot(plt)
plt.clf()

st.markdown("""---""")
st.title("Вины дороже 200$")

plt.figure(figsize=(20,5))
plt.title("Distribution of price")
ax1 = sns.histplot(data[data["price"]<200]['price'])

percent=data[data['price']>200].shape[0]/data.shape[0]*100
st.pyplot(plt)
plt.clf()

st.markdown("""---""")
st.title("Цены на вино в разных странах")
z=data.groupby(['country'])['price','points'].mean().reset_index().sort_values('price',ascending=False)
z[['country','price']].head(n=10)
plt.figure(figsize = (14,6))
plt.title('Wine prices in diffrent countries')
sns.barplot(x = 'country', y="price", data=z.head(10))
locs, labels = plt.xticks()
plt.show()

st.pyplot(plt)
plt.clf()

st.markdown("""---""")
st.title("Баллы за вино в разных странах")

z=z.sort_values('points', ascending=False)
z[['country','points']].head(10)
plt.figure(figsize = (14,6))
plt.title('Points for wines in diffrent countries')
sns.set_color_codes("pastel")
sns.barplot(x = 'country', y="points", data=z.head(5))
locs, labels = plt.xticks()
plt.show()
st.pyplot(plt)
plt.clf()

country=data['country'].value_counts()
country.head(10).plot.bar()
country.head(20)
st.pyplot(plt)
plt.clf()

st.markdown("""---""")
z['quality/price']=z['points']/z['price']
z.sort_values('quality/price', ascending=False)[['country','quality/price']]


st.title("Исследование данных с помощью прямоугольных графиков")


df1= data[data.variety.isin(data.variety.value_counts().head(6).index)]

plt.figure(figsize = (14,6))
sns.boxplot(
    x = 'variety',
    y = 'points',
    data = df1
)

st.pyplot(plt)
plt.clf()


X=data.drop(columns=['points'])

X=X.fillna(-1)
categorical_features_indices =[0,1, 3,4,5,6,7,8,9,10]
y=data['points']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2,
                                                    random_state=52)


def perform_model(X_train, y_train, X_valid, y_valid, X_test, y_test):
    model = CatBoostRegressor(
        random_seed=400,
        loss_function='RMSE',
        iterations=400,
    )

    model.fit(
        X_train, y_train,
        cat_features=categorical_features_indices,
        eval_set=(X_valid, y_valid),
        verbose=False
    )

    print("RMSE on training data: " + model.score(X_train, y_train).astype(str))
    print("RMSE on test data: " + model.score(X_test, y_test).astype(str))

    return model

model=perform_model(X_train, y_train,X_valid, y_valid,X_test, y_test)

feature_score = pd.DataFrame(list(zip(X.dtypes.index, model.get_feature_importance(Pool(X, label=y, cat_features=categorical_features_indices)))),
                columns=['Feature','Score'])

feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')

plt.rcParams["figure.figsize"] = (12,7)
ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)
ax.set_xlabel('')

rects = ax.patches

labels = feature_score['Score'].round(2)

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')

plt.show()

st.markdown("""---""")
st.title("Рейтинг важности функции Catboost")


st.pyplot(plt)
plt.clf()
