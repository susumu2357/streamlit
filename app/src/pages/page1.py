import streamlit as st
from ..utils import Page
import pandas as pd
import numpy as np
import io
import gzip
import pickle
import requests
import altair as alt
import pydeck as pdk

def load_pickle(url):
    response = requests.get(url)
    gzip_file = io.BytesIO(response.content)
    with gzip.open(gzip_file, 'rb') as f:
        return pickle.load(f)

url = 'https://storage.googleapis.com/test_bucket_20200914/year3_with_location_2015_2019.pkl.gz'

# load
school_df_dict = load_pickle(url)

@st.cache
def load_df():
    df_year = {}
    commun_year = {}

    for year in ['2015', '2016', '2017', '2018', '2019']:
        df1 = school_df_dict[year]['Math']
        df2 = school_df_dict[year]['Swedish']

        df1['achievement_percentage'] = df1['achievement_percentage'].replace('.', '0').replace('..', '0').astype(float)
        df2['achievement_percentage'] = df2['achievement_percentage'].replace('.', '0').replace('..', '0').astype(float)

        df_year[year] = df1.groupby(['school', 'school_commun', 'type', 'nr_students', 'formatted_address', 'latitude', 'longitude'])['achievement_percentage'].median().reset_index()\
            .merge(df2.groupby('school')['achievement_percentage'].median().reset_index(), on='school')
        df_year[year] = df_year[year].rename(columns={'achievement_percentage_x':'math_achievement_percentage', 'achievement_percentage_y':'swedish_achievement_percentage'})
        df_year[year]['achievement_percentage'] = df_year[year][['math_achievement_percentage', 'swedish_achievement_percentage']].mean(axis=1)
    
        commun_year[year] = pd.DataFrame(df_year[year].groupby(['school_commun', 'type'])['achievement_percentage']\
            .describe().sort_values('50%', ascending=False).to_records())
        commun_year[year]['school_commun_type'] = [a+'_'+b for a,b in zip(commun_year[year]['school_commun'].values, commun_year[year]['type'].values)]
        commun_year[year]['rank'] = commun_year[year].index + 1

    time_seires_df = pd.concat(
        [commun_year['2015'][['school_commun_type', 'rank']], commun_year['2016'][['school_commun_type', 'rank']]],
        ignore_index=True)
    for year in ['2017', '2018', '2019']:
        time_seires_df = pd.concat(
            [time_seires_df, commun_year[year][['school_commun_type', 'rank']]],
            ignore_index=True)
    year_list = []
    for year in ['2015', '2016', '2017', '2018', '2019']:
        year_list += [year]*len(commun_year[year])
    time_seires_df = pd.concat(
        [time_seires_df, pd.Series(year_list, name='year')],
        axis=1)

    return df_year, commun_year, time_seires_df

class Page1(Page):
    def __init__(self, state):
        self.state = state

    def write(self):
        df_year, commun_year, time_seires_df = load_df()

        st.title('Elementary School Grade 3')
        st.header("Statistics of commun")
        
        # interval selection in the scatter plot
        pts = alt.selection(type="interval", encodings=["x"])

        # left panel: scatter plot
        time_series = alt.Chart().mark_line().encode(
            x='year:O',
            y='rank:O',
            color='school_commun_type',
        )

        points = time_series.mark_point(color='black').encode(
            x='year:O',
            y='rank:O',
            size=alt.value(100),
            tooltip=['school_commun_type', 'rank', 'year']
        )

        points = points.add_selection(alt.selection_single())

        left = (time_series + points).transform_filter(
            pts
        ).properties(width=400, height=400)

        # right panel: histogram
        mag = alt.Chart().mark_bar().encode(
            x='rank_bin:N',
            y="count()",
            color=alt.condition(pts, alt.value("lightgreen"), alt.value("lightgray"))
        ).properties(
            width=300,
            height=200
        ).add_selection(pts)

        # build the chart:
        ranking = alt.hconcat(
            left,
            mag,
            data=time_seires_df
        ).transform_bin(
            'rank_bin',
            field='rank',
            bin=alt.Bin(maxbins=26)
        )

        st.subheader('Select interval on the right chart')
        st.write(ranking)

        st.subheader('Achievement percentage in average of Math and Swedish')
        option_year = st.selectbox(
        'Choose year',
        ['2015', '2016', '2017', '2018', '2019'])

        bars = alt.Chart(commun_year[option_year]).mark_bar().encode(
            x='50%:Q',
            y=alt.Y('school_commun_type', sort=None),
            color='type',
            tooltip=['count', '50%', 'mean', 'std']
        )

        text = bars.mark_text(
            align='left',
            baseline='middle',
            dx=3
        ).encode(
            text='50%:Q'
        )

        st.write((bars + text).properties(width=600))
        st.write(commun_year[option_year][commun_year[option_year].columns[:-1]])

        st.header("Statistics of commun")
        st.subheader('Achievement percentage of each school in the selected commun')
        option_commun = st.selectbox(
        'Choose commun',
        df_year[option_year]['school_commun'].sort_values().unique())

        commun_df = df_year[option_year][df_year[option_year]['school_commun']==option_commun]
        commun_df = commun_df.sort_values('achievement_percentage', ascending=False)
        
        st.subheader(f"Statistics of {option_commun}")

        # Define a layer to display on a map
        layer = pdk.Layer(
            "ScatterplotLayer",
            commun_df,
            pickable=True,
            opacity=0.5,
            stroked=True,
            filled=True,
            auto_highlight=True,
            radius_scale=3,
            radius_min_pixels=2,
            radius_max_pixels=25,
            line_width_min_pixels=1,
            get_position=['longitude', 'latitude'],
            get_radius='achievement_percentage',
            get_fill_color=[255, 140, 0],
            get_line_color=[0, 0, 0],
        )

        # Set the viewport location
        view_state = pdk.ViewState(
            latitude=commun_df['latitude'].mean(), 
            longitude=commun_df['longitude'].mean(), 
            zoom=10, bearing=0, pitch=0)

        # Render
        r = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            layers=[layer], initial_view_state=view_state, 
            tooltip={"text": "{school}\n{formatted_address}\nachievement_percentage:{achievement_percentage}"})

        st.pydeck_chart(r)

        base = alt.Chart(commun_df).encode(y=alt.Y('school', sort=None),)

        bar = base.mark_bar().encode(x='achievement_percentage:Q',
                                    color='type',
                                    tooltip=[
                                        'achievement_percentage',
                                        'math_achievement_percentage',
                                        'swedish_achievement_percentage',
                                        'nr_students'
                                            ]
        )

        math =  base.mark_point(color='green').encode(
            x='math_achievement_percentage:Q',
        )

        swedish =  base.mark_point(color='red').encode(
            x='swedish_achievement_percentage:Q',
        )

        text = base.mark_text(
            align='left',
            baseline='middle',
        ).encode(
            text='achievement_percentage:Q'
        )

        st.write((bar + math + swedish + text).properties(width=600))
        st.write(commun_df[['school', 'school_commun', 'type', 'nr_students', 
        'math_achievement_percentage', 'swedish_achievement_percentage', 'achievement_percentage']])




