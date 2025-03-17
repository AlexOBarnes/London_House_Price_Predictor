'''Functions that process requests from streamlit app'''
from os import environ as ENV
from dotenv import load_dotenv
from opencage.geocoder import OpenCageGeocode
import streamlit as st


@st.cache_data
def get_lat_lon(postcode):
    geocoder = OpenCageGeocode(ENV['MAPS_API_KEY'])
    result = geocoder.geocode(postcode)
    if result:
        return result[0]['geometry']['lat'], result[0]['geometry']['lng']
    return None


if __name__ == '__main__':
    load_dotenv()
    print(get_lat_lon('SW1A 1AA'))
