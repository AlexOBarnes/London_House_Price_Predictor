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
from utils import get_lat_lon, display_commute_times


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


def target_location_input_box():
    if 'targets' not in st.session_state:
        st.session_state.targets = []
    return st.text_input(
        'Enter an address or list of addresses separated by ";" that you want to measure this properties commute to:', placeholder='Address')


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
    if postcode in st.session_state.postcode_options:
        values = [
            float(predictions["lower_bound"]),
            float(predictions["middle_prediction"]),
            float(predictions["upper_bound"])
        ]
    else:
        values = [
            float(predictions["lower_bound"]) - 150000,
            float(predictions["middle_prediction"]) - 150000,
            float(predictions["upper_bound"]) - 150000
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

    st.markdown(
        "<h4 style='text-align: center; color: #8AB4F8;'>🏡 Our model predicts that 80% of properties of this type and size fall within the following range:</h4>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([4, 6, 4])
    with col2:
        st.markdown(
            f"<h2 style='text-align: center; background-color: #1E1E1E; color: #4CAF50; padding: 12px; border-radius: 10px;'>"
            f"💰 £{values[0]:,.2f} → £{values[2]:,.2f} 💰</h2>",
            unsafe_allow_html=True
        )

    st.markdown(
        "<h4 style='text-align: center; color: #FBC02D;'>📍 With a median value of:</h4>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([3, 4, 3])
    with col2:
        st.markdown(
            f"<h2 style='text-align: center; background-color: #292929; color: #FBC02D; padding: 12px; border-radius: 10px;'>"
            f"🏷️ £{values[1]:,.2f}</h2>",
            unsafe_allow_html=True
        )


def map_targets(home_lat, home_long):
    target_dict = {'lat': [home_lat], 'lon': [home_long], 'size': [200]}
    for target in st.session_state.targets:
        lat, lon = get_lat_lon(target)
        target_dict['lat'] += [lat]  # Add the new lat to the list
        target_dict['lon'] += [lon]  # Add the new lon to the list
        target_dict['size'] += [100]

    targets = pd.DataFrame(target_dict)
    st.session_state.target_df = targets
    st.map(targets, size='size', zoom=12)


def calculate_commutes(lat, long):
    if st.session_state.targets:
        map_targets(lat, long)
        display_commute_times(lat, long)
        ...


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
        target = target_location_input_box()
        if confirmation():
            postcode = st.session_state.get('postcode', None)
            latitude, longitude = get_lat_lon(postcode)
            bathrooms = st.session_state.get('bathrooms', None)
            bedrooms = st.session_state.get('bedrooms', None)
            livingrooms = st.session_state.get('livingrooms', None)
            units = st.session_state.get('units', 'Square Metres')
            area = st.session_state.get(
                'area', 0.0) * (1-((st.session_state.get('units') == 'Square Feet') * 0.907097))
            tenure = st.session_state.get('tenure', 'Leasehold')
            property_type = st.session_state.get('property_type', None)
            energy_rating = st.session_state.get('energy_rating', None)
            if target:
                st.session_state.targets = target.split(';')
            else:
                st.session_state.targets = [ENV['DEFAULT_LOCATION']]
            map_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
            st.map(map_data, zoom=12, size=100)
            predictions = predict_price_range(area, tenure, property_type,
                                              energy_rating, bathrooms, bedrooms, livingrooms, postcode)
            plot = st.container()
            with plot:
                plot_prediction(predictions, bedrooms, property_type, postcode)
            calculate_commutes(latitude, longitude)
    # TODO: Input box for commute location, output for commute time
    # TODO: Scrape of Rightmove for postcode, bathrooms, bedrooms, livingrooms, area, tenure, property_type, energy_rating. Give option to fill in missing values (DataEditor)
