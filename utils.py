'''Functions that process requests from streamlit app'''
from os import environ as ENV
import json
import httpx
import pandas as pd
from dotenv import load_dotenv
from opencage.geocoder import OpenCageGeocode
import streamlit as st
from time import sleep


def get_commute_times(start_lat, start_lon, dest_row) -> int:
    q = f'origin={start_lat},{start_lon}&destination={dest_row.iloc[0]},{dest_row.iloc[1]}'
    url = st.session_state.base_url + q
    with httpx.Client() as client:
        res = client.get(url)
    res_json = res.json()
    st.write(res_json)
    sleep(120)


def display_commute_times(lat, lon):
    construct_url_commute()
    for i, row in st.session_state.target_df.iterrows():
        get_commute_times(lat, lon, row)


@st.cache_data
def get_lat_lon(address: str):
    geocoder = OpenCageGeocode(ENV['MAPS_API_KEY'])
    if address:
        result = geocoder.geocode(address)
    if result:
        return result[0]['geometry']['lat'], result[0]['geometry']['lng']
    return None


@st.cache_data
def construct_url_commute():
    '''Creates the url that will be used for future queries'''
    st.session_state.base_url = f"https://router.here.com/v8/routes?apiKey={ENV['COMMUTE_API_KEY']}&"


if __name__ == '__main__':
    load_dotenv()
    print(get_lat_lon('SW1A 1AA'))
