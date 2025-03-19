'''Streamlit app that takes in housing values and returns an estimated price'''
from os import environ as ENV
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle as pk
import streamlit as st
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from utils import get_lat_lon


def load_session_state():
    '''Loads in models for predictions'''
    st.session_state.postcode_options = pd.read_pickle(
        'data/house_df.pkl')['postcode'].unique().tolist()
    st.session_state.model_list = load_models()
    st.session_state.processor = get_standard_processor()
    st.session_state.sample_data = pd.read_pickle('data/house_df.pkl')[['tenure', 'propertyType', 'currentEnergyRating', 'bathrooms',
                                                                        'bedrooms', 'livingRooms', 'floorAreaSqM', 'postcode']]
    st.session_state.processor.fit(st.session_state.sample_data)


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
        st.text_input(
            'Postcode:', key='postcode', placeholder="Select postcode...")
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


def predict_price_range(floor_area, tenure, property_type, energy_rating, bathrooms, bedrooms, living_rooms, postcode):
    """Processes property details using the cached preprocessor and predicts the price using cached models."""
    models = load_models()
    column_names = ['floorAreaSqM', 'tenure', 'propertyType', 'currentEnergyRating',
                    'bathrooms', 'bedrooms', 'livingRooms', 'postcode']
    input_values = [floor_area, tenure, property_type,
                    energy_rating, bathrooms, bedrooms, living_rooms, postcode]
    input_data = pd.DataFrame([input_values], columns=column_names)
    middle_pred = models[0].predict(input_data)
    lower_pred = models[1].predict(input_data)
    upper_pred = models[2].predict(input_data)
    return {
        "lower_bound": lower_pred,
        "middle_prediction": middle_pred,
        "upper_bound": upper_pred
    }


def plot_prediction(predictions: dict, bedroom: str, property_type: str, postcode: str):
    values = [
        float(predictions["lower_bound"]) - 150000,
        float(predictions["middle_prediction"]),
        float(predictions["upper_bound"]) + 150000
    ]
    labels = ["Lower Bound", "Middle Prediction", "Upper Bound"]
    bar_colours = ['#4CAF50', '#FFC107', '#F44336']

    fig, ax = plt.subplots(figsize=(5, 2))

    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    ax.bar(labels, values, color=bar_colours, edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Cost (£)', fontsize=10, color='#FFFFFF')
    ax.set_xlabel("Price Prediction Bounds", fontsize=10, color='#FFFFFF')
    ax.set_title(f'Estimated price bounds of {bedroom} Bedroom {property_type} at {postcode}',
                 fontsize=11, fontweight='bold', color='#FFFFFF')

    ax.tick_params(axis='both', colors='#EEEEEE', labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    st.pyplot(fig)

    st.write('Our model predicts that 80% of properties of this type and size fall within the following range:')
    col1, col2, col3 = st.columns([5, 5, 3])
    with col2:
        st.markdown(f'## £{values[0]:.2f} -> £{values[2]:.2f}')

    st.write('With a median value of :')
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        st.markdown(f'## £{values[1]:.2f}')


@st.cache_resource
def load_models() -> list[RandomForestRegressor]:
    '''Returns a list of random forest models'''
    models = []
    for file in ['model_cache/RandomForest.pkl', 'model_cache/lowerRandomForest.pkl', 'model_cache/upperRandomForest.pkl']:
        with open(file, 'rb') as f:
            models.append(pk.load(f))
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
    load_dotenv()
    st.set_page_config(layout="wide")
    load_session_state()
    title_bar()
    choice = input_control()
    if choice == 'URL':
        url_input_box()
        if confirmation():
            ...
    if choice == 'Properties':
        input_boxes()
        if confirmation():
            postcode = st.session_state.get('postcode', None)
            latitude, longitude = get_lat_lon(postcode=postcode)
            bathrooms = st.session_state.get('bathrooms', None)
            bedrooms = st.session_state.get('bedrooms', None)
            livingrooms = st.session_state.get('livingrooms', None)
            units = st.session_state.get('units', 'Square Metres')
            area = st.session_state.get(
                'area', 0.0) * (1-((st.session_state.get('units') == 'Square Feet') * 0.907097))
            tenure = st.session_state.get('tenure', 'Leasehold')
            property_type = st.session_state.get('property_type', None)
            energy_rating = st.session_state.get('energy_rating', None)
            map_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
            st.map(map_data, zoom=12, size=100)
            predictions = predict_price_range(area, tenure, property_type,
                                              energy_rating, bathrooms, bedrooms, livingrooms, postcode)
            plot = st.container()
            with plot:
                plot_prediction(predictions, bedrooms, property_type, postcode)
    # TODO: Map for location based off postcode
    # TODO: Inference with model
    # TODO: Input box for commute location, output for commute time
    # TODO: Scrape of Rightmove for postcode, bathrooms, bedrooms, livingrooms, area, tenure, property_type, energy_rating. Give option to fill in missing values (DataEditor)
