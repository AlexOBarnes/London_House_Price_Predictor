'''Streamlit app that takes in housing values and returns an estimated price'''
import pickle as pk
import streamlit as st
import polars as pl


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


if __name__ == '__main__':
    title_bar()
