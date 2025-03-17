'''Streamlit app that takes in housing values and returns an estimated price'''
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle as pk
import streamlit as st
import polars as pl
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_session_state():
    '''Loads in models for predictions'''
    st.session_state.postcode_options = pd.read_pickle(
        'data/house_df.pkl')['postcode'].unique().tolist()
    st.session_state.model_list = load_models()
    st.session_state.processor = get_standard_processor()


def title_bar():
    '''Defines the orientation of the title bar of this streamlit app'''
    col1, col2, col3 = st.columns([3, 2.5, 2.5])
    with col2:
        st.title('House Price Evaluator')
    st.divider()


def input_control():
    '''Defines a segmented_control box'''
    options = ['URL', 'Properties']
    col1, col2 = st.columns([2, 4])
    with col1:
        return st.segmented_control('Input', options=options, selection_mode='single', default='Properties')


def url_input_box():
    '''Defines the orientation of the input boxes and following process of scraping/storing data'''
    st.warning('Section is in Production...')
    st.text_input('Rightmove Property URL:', placeholder='URL')


def input_boxes():
    '''Defines the orientation of the input boxes for entering house features'''
    col8, col1, col2, col3, col4, col5, col6, col7 = st.columns(8)
    with col8:
        st.selectbox(
            'Postcode:', options=st.session_state.postcode_options, key='postcode', index=None, placeholder="Select postcode...")
    with col1:
        st.slider(
            'Bathrooms:', 0, 20, st.session_state.get('bathrooms', 1), key='bathrooms'
        )

    with col2:
        st.slider(
            'Bedrooms:', 0, 20, st.session_state.get('bedrooms', 1), key='bedrooms'
        )

    with col3:
        st.slider(
            'Living Rooms:', 0, 20, st.session_state.get('livingrooms', 1), key='livingrooms'
        )

    with col4:
        units = st.selectbox(
            'Units', ('Square Metres', 'Square Feet'), key='units')
        area = st.number_input(
            'Area:', value=st.session_state.get('area', 0.0), key='area')

    with col5:
        st.radio(
            'Tenure', ['Leasehold', 'Freehold'], index=0, key='tenure'
        )

    with col6:
        property_options = [
            'Purpose Built Flat', 'Flat/Maisonette', 'End Terrace House', 'Mid Terrace House',
            'Terrace Property', 'Semi-Detached House', 'Converted Flat', 'Detached House',
            'Terraced', 'Bungalow Property', 'Detached Property', 'Terraced Bungalow',
            'Mid Terrace Property', 'Detached Bungalow', 'Semi-Detached Bungalow',
            'Mid Terrace Bungalow', 'Semi-Detached Property', 'End Terrace Property',
            'End Terrace Bungalow'
        ]
        st.selectbox(
            'Property Type:', options=property_options, key='property_type', index=None, placeholder="Select type..."
        )

    with col7:
        st.selectbox(
            'Energy Rating', options=['A', 'B', 'C', 'D', 'E', 'F', 'G'], key='energy_rating', index=None, placeholder="Select rating..."
        )


def confirmation():
    return st.button('Start Process', type='primary')


@st.cache_resource
def load_models() -> list[RandomForestRegressor]:
    '''Returns a list of random forest models'''
    models = []
    for file in ['model_cache/RandomForest.pkl', 'model_cache/lowerRandomForest.pkl', 'model_cache/upperRandomForest.pkl']:
        with open(file, 'rb') as f:
            models.append(pk.load(f))
    st.session_state.middle_model = models[0]
    st.session_state.lower_model = models[1]
    st.session_state.upper_model = models[2]
    return models


@st.cache_resource
def get_standard_processor():
    numerical_features = ['floorAreaSqM']
    categorical_features = ['tenure', 'propertyType', 'currentEnergyRating', 'bathrooms',
                            'bedrooms', 'livingRooms', 'postcode']

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])
    st.session_state.processor = preprocessor
    return preprocessor


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    load_session_state()
    title_bar()
    choice = input_control()
    if choice == 'URL':
        url_input_box()
    if choice == 'Properties':
        input_boxes()
    if confirmation():
        postcode = st.session_state.get('postcode', None)
        bathrooms = st.session_state.get('bathrooms', None)
        bedrooms = st.session_state.get('bedrooms', None)
        livingrooms = st.session_state.get('livingrooms', None)
        units = st.session_state.get('units', 'Square Metres')
        area = st.session_state.get(
            'area', 0.0) * (1-((st.session_state.get('units') == 'Square Feet') * 0.907097))
        tenure = st.session_state.get('tenure', 'Leasehold')
        property_type = st.session_state.get('property_type', None)
        energy_rating = st.session_state.get('energy_rating', None)
        st.write(f"Postcode: {postcode}")
        st.write(f"Bathrooms: {bathrooms}")
        st.write(f"Bedrooms: {bedrooms}")
        st.write(f"Living Rooms: {livingrooms}")
        st.write(f"Units: {units}")
        st.write(f"Area: {area}")
        st.write(f"Tenure: {tenure}")
        st.write(f"Property Type: {property_type}")
        st.write(f"Energy Rating: {energy_rating}")
    # TODO: Map for location based off postcode
    # TODO: Inference with model
    # TODO: Input box for commute location, output for commute time
    # TODO: Scrape of Rightmove for postcode, bathrooms, bedrooms, livingrooms, area, tenure, property_type, energy_rating. Give option to fill in missing values (DataEditor)
