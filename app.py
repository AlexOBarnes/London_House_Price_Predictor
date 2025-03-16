'''Streamlit app that takes in housing values and returns an estimated price'''
from sklearn.ensemble import RandomForestRegressor
import pickle as pk
import streamlit as st
import polars as pl
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_session_state():
    '''Loads in models for predictions'''
    ...


def title_bar():
    '''Defines the orientation of the title bar of this streamlit app'''
    ...


def input_box():
    '''Defines the orientation of the input boxes for entering house features'''
    ...


def url_input_box():
    '''Defines the orientation of the input boxes and following process of scraping/storing data'''
    ...


@st.cache_resource
def load_models() -> list[RandomForestRegressor]:
    '''Returns a list of gra'''
    models = []
    for file in ['model_cache\RandomForest.pkl', 'model_cache\lowerRandomForest.pkl',
                 'model_cache\upperRandomForest.pkl']:
        with open(file, 'r') as f:
            models.append(pk.load(f))
    st.session_state.middle_model = models[0]
    st.session_state.lower_model = models[1]
    st.session_state.upper_model = models[2]
    st.session_state.model_list = models
    return models


@st.cache_resource
def get_standard_processor():
    numerical_features = ['bathrooms',
                          'bedrooms', 'floorAreaSqM', 'livingRooms']
    categorical_features = ['tenure', 'propertyType', 'currentEnergyRating']

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])
    st.session_state.processor = preprocessor
    return preprocessor


if __name__ == '__main__':
    load_models()
    title_bar()
