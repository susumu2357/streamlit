import streamlit as st
from ..utils import Page
import pandas as pd
import numpy as np
import io
import gzip
import pickle
import requests
import altair as alt

def load_pickle(url):
    response = requests.get(url)
    gzip_file = io.BytesIO(response.content)
    with gzip.open(gzip_file, 'rb') as f:
        return pickle.load(f)

url = 'https://storage.googleapis.com/test_bucket_20200914/year3_2017_2019.pkl.gz'

# load
school_df_dict = load_pickle(url)

class Page1(Page):
    def __init__(self, state):
        self.state = state

    def write(self):
        title = "Math"

        st.title(title)
        option = st.selectbox(
        'Choose year',
        ['2017', '2018', '2019'])

        df = school_df_dict[option][title]

        commun_type_df = pd.DataFrame(df.groupby(['school_commun', 'type'])['achievement_percentage']\
            .describe().sort_values('50%', ascending=False).to_records())
        commun_type_df['school_commun_type'] = [a+'_'+b for a,b in zip(commun_type_df['school_commun'].values, commun_type_df['type'].values)]

        st.write("Statistics of communs")
        st.write(alt.Chart(commun_type_df).mark_line().encode(
            x=alt.X('school_commun_type', sort=None),
            y='50%',)
            # commun_type_df[['school_commun_type', '50%']].set_index('school_commun_type')
            )
        st.write(commun_type_df[commun_type_df.columns[:-1]])

        option_commun = st.selectbox(
        'Choose commun',
        df['school_commun'].unique())

        commun_df = df[df['school_commun']==option_commun]

        st.write(f"Statistics of {option_commun}")
        st.write(alt.Chart(commun_df).mark_circle().encode(
            x=alt.X('school', sort=None),
            y='achievement_percentage',
            color='sub',
            size='nr_students'
            ))
        st.write(commun_df)




