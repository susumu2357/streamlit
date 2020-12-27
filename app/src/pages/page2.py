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


url = 'https://storage.googleapis.com/test_bucket_20200914/year6_with_location_2015_2019.pkl.gz'

# load
school_df_dict = load_pickle(url)


@st.cache
def load_df():
    df_year = {}
    commun_year = {}

    for year in ['2015', '2016', '2017', '2018', '2019']:
        df1 = school_df_dict[year]['English']
        df2 = school_df_dict[year]['Math']
        df3 = school_df_dict[year]['Swedish']

        tmp_df = df1.groupby(['school', 'school_commun', 'type', 'formatted_address', 'latitude', 'longitude'])['average_score'].median().reset_index()\
            .merge(df2.groupby('school')['average_score'].median().reset_index(), on='school')
        tmp_df = tmp_df.merge(df3.groupby('school')[
                              'average_score'].median().reset_index(), on='school')
        df_year[year] = tmp_df.rename(columns={
                                      'average_score_x': 'english_score', 'average_score_y': 'math_score', 'average_score': 'swedish_score'})
        df_year[year]['average_score'] = df_year[year][[
            'english_score', 'math_score', 'swedish_score']].mean(axis=1).round(2)

        commun_year[year] = pd.DataFrame(df_year[year].groupby(['school_commun', 'type'])['average_score']
                                         .describe().sort_values('50%', ascending=False).to_records())
        commun_year[year]['school_commun_type'] = [a+'_'+b for a, b in zip(
            commun_year[year]['school_commun'].values, commun_year[year]['type'].values)]
        commun_year[year]['rank'] = commun_year[year].index + 1

    time_seires_df = pd.concat(
        [commun_year['2015'][['school_commun_type', 'rank']],
            commun_year['2016'][['school_commun_type', 'rank']]],
        ignore_index=True)
    for y in ['2017', '2018', '2019']:
        time_seires_df = pd.concat(
            [time_seires_df, commun_year[y][['school_commun_type', 'rank']]], ignore_index=True)
    year_list = []
    for year in ['2015', '2016', '2017', '2018', '2019']:
        year_list += [year]*len(commun_year[year])
    time_seires_df = pd.concat(
        [time_seires_df, pd.Series(year_list, name='year')], axis=1)

    school_time_seires_df = pd.DataFrame(
        columns=list(df_year['2015'].columns)
        + ['year', 'english_rank', 'math_rank', 'swedish_rank', 'average_rank'])
    for year in ['2015', '2016', '2017', '2018', '2019']:
        tmp_df = df_year[year]
        tmp_df['year'] = [year]*len(tmp_df)
        tmp_df.sort_values('english_score', ascending=False, inplace=True)
        tmp_df['english_rank'] = tmp_df.reset_index().index + 1
        tmp_df.sort_values('math_score', ascending=False, inplace=True)
        tmp_df['math_rank'] = tmp_df.reset_index().index + 1
        tmp_df.sort_values('swedish_score', ascending=False, inplace=True)
        tmp_df['swedish_rank'] = tmp_df.reset_index().index + 1
        tmp_df.sort_values('average_score', ascending=False, inplace=True)
        tmp_df['average_rank'] = tmp_df.reset_index().index + 1
        school_time_seires_df = school_time_seires_df.append(tmp_df)
    school_time_seires_df.reset_index(inplace=True)

    return df_year, commun_year, time_seires_df, school_time_seires_df


class Page2(Page):
    def __init__(self, state):
        self.state = state

    def write(self):
        df_year, commun_year, time_seires_df, school_time_seires_df = load_df()

        st.title('Elementary School Grade 6')

        st.header("Statistics of school")

        school_option = st.selectbox(
            'Choose subject',
            ['average_rank', 'english_rank', 'math_rank', 'swedish_rank'])

        school_top = st.slider('Choose top-N',
                               min_value=1, max_value=50, value=10, step=1)

        # st.selectbox(
        #     'Choose top-N',
        #     [5, 10, 30])

        top_df = school_time_seires_df[school_time_seires_df[school_option] <= school_top].sort_values(
            school_option, ascending=True)
        school_score_option = school_option.replace('rank', 'score')
        top_df_avg = pd.DataFrame(top_df.groupby(['school', 'school_commun', 'type', 'formatted_address', 'latitude', 'longitude'])[
                                  ['average_score', 'english_score', 'math_score', 'swedish_score']].mean().round(2).to_records()).sort_values(school_score_option, ascending=False)

        school_time_series = alt.Chart(top_df).mark_line().encode(
            x='year:O',
            y=alt.Y(f'{school_option}:O', sort=None),
            color='school',
        )

        school_points = school_time_series.mark_point(color='black').encode(
            x='year:O',
            y=alt.Y(f'{school_option}:O', sort=None),
            size=alt.value(100),
            tooltip=['school', 'school_commun',
                     'type', f'{school_option}', 'year']
        )

        st.write(
            (school_time_series + school_points).properties(width=600, height=400)
        )

        st.subheader(f'5 years average of top-{school_top} schools')

        base = alt.Chart(top_df_avg).encode(y=alt.Y('school', sort=None),)

        bar = base.mark_bar().encode(x=f'{school_score_option}:Q',
                                     color='type',
                                     tooltip=[
                                         'school_commun',
                                         'average_score',
                                         'english_score',
                                         'math_score',
                                         'swedish_score',
                                     ]
                                     )

        english = base.mark_point(color='blue').encode(
            x='english_score:Q',
        )

        math = base.mark_point(color='green').encode(
            x='math_score:Q',
        )

        swedish = base.mark_point(color='red').encode(
            x='swedish_score:Q',
        )

        text = base.mark_text(
            align='left',
            baseline='middle',
        ).encode(
            text=f'{school_score_option}:Q'
        )

        st.write((bar + english + math + swedish + text).properties(width=600))

        # Define a layer to display on a map
        layer = pdk.Layer(
            "ScatterplotLayer",
            top_df_avg,
            pickable=True,
            opacity=0.5,
            stroked=True,
            filled=True,
            auto_highlight=True,
            radius_scale=10,
            # radius_min_pixels=5,
            # radius_max_pixels=25,
            line_width_min_pixels=1,
            get_position=['longitude', 'latitude'],
            get_radius=20,
            get_fill_color=[255, 140, 0],
            get_line_color=[0, 0, 0],
        )

        # Set the viewport location
        view_state = pdk.ViewState(
            latitude=top_df['latitude'].mean(),
            longitude=top_df['longitude'].mean(),
            zoom=9, bearing=0, pitch=0)

        # Render
        r = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            layers=[layer], initial_view_state=view_state,
            tooltip={"text": "{school}\n{school_commun}\nenglish_score:{english_score}\nmath_score:{math_score}\nswedish_score:{swedish_score}\naverage_score:{average_score}"})

        st.pydeck_chart(r)

        st.write(top_df[['school', 'school_commun', 'type', 'year', 'average_score',
                         'average_rank', 'english_rank', 'math_rank', 'swedish_rank']])

        st.header("Statistics of commun")

        top = st.slider('Choose top-N',
                        min_value=1, max_value=len(time_seires_df['school_commun_type'].unique()), value=10, step=1)

        time_series = alt.Chart(time_seires_df[time_seires_df['rank'] <= top]).mark_line().encode(
            x='year:O',
            y=alt.Y('rank:O', sort=None),
            color='school_commun_type',
        )

        points = time_series.mark_point(color='black').encode(
            x='year:O',
            y='rank:O',
            size=alt.value(100),
            tooltip=['school_commun_type', 'rank', 'year']
        )

        st.write(
            (time_series + points).properties(width=600, height=400)
        )

        # # interval selection in the scatter plot
        # pts = alt.selection(type="interval", encodings=["x"])

        # # left panel: scatter plot
        # time_series = alt.Chart().mark_line().encode(
        #     x='year:O',
        #     y='rank:O',
        #     color='school_commun_type',
        # )

        # points = time_series.mark_point(color='black').encode(
        #     x='year:O',
        #     y='rank:O',
        #     size=alt.value(100),
        #     tooltip=['school_commun_type', 'rank', 'year']
        # )

        # points = points.add_selection(alt.selection_single())

        # left = (time_series + points).transform_filter(
        #     pts
        # ).properties(width=400, height=400)

        # # right panel: histogram
        # mag = alt.Chart().mark_bar().encode(
        #     x='rank_bin:N',
        #     y="count()",
        #     color=alt.condition(pts, alt.value(
        #         "lightgreen"), alt.value("lightgray"))
        # ).properties(
        #     width=300,
        #     height=200
        # ).add_selection(pts)

        # # build the chart:
        # ranking = alt.hconcat(
        #     left,
        #     mag,
        #     data=time_seires_df
        # ).transform_bin(
        #     'rank_bin',
        #     field='rank',
        #     bin=alt.Bin(maxbins=26)
        # )

        # st.subheader('Select interval on the right chart')
        # st.write(ranking)

        st.subheader('Average score (0 to 20) of English, Math and Swedish')
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
        st.write(commun_year[option_year]
                 [commun_year[option_year].columns[:-1]])

        st.subheader(
            'Average score of each school located in the selected commun')
        option_commun = st.multiselect(
            label='Choose commun',
            options=list(df_year[option_year]
                         ['school_commun'].sort_values().unique()),
            default=['Danderyd', 'Stockholm'],
        )

        commun_df = df_year[option_year][df_year[option_year]
                                         ['school_commun'].str.contains('|'.join(option_commun))]
        commun_df = commun_df.sort_values('average_score', ascending=False)
        commun_df['school_with_commun'] = [a+'_'+b for a, b in zip(
            commun_df['school'].values, commun_df['school_commun'].values)]

        st.subheader(f"Statistics of {', '.join(option_commun)}")

        # Define a layer to display on a map
        layer = pdk.Layer(
            "ScatterplotLayer",
            commun_df,
            pickable=True,
            opacity=0.5,
            stroked=True,
            filled=True,
            auto_highlight=True,
            radius_scale=10,
            radius_min_pixels=5,
            radius_max_pixels=25,
            line_width_min_pixels=1,
            get_position=['longitude', 'latitude'],
            get_radius='average_score',
            get_fill_color=[255, 140, 0],
            get_line_color=[0, 0, 0],
        )

        # Set the viewport location
        view_state = pdk.ViewState(
            latitude=commun_df['latitude'].mean(),
            longitude=commun_df['longitude'].mean(),
            zoom=9, bearing=0, pitch=0)

        # Render
        r = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            layers=[layer], initial_view_state=view_state,
            tooltip={"text": "{school}\n{formatted_address}\naverage_score:{average_score}"})

        st.pydeck_chart(r)

        base = alt.Chart(commun_df).encode(
            y=alt.Y('school_with_commun', sort=None),)

        bar = base.mark_bar().encode(x='average_score:Q',
                                     color='school_commun',
                                     tooltip=[
                                         'school',
                                         'average_score',
                                         'english_score',
                                         'math_score',
                                         'swedish_score',
                                     ]
                                     )

        english = base.mark_point(color='blue').encode(
            x='english_score:Q',
        )

        math = base.mark_point(color='green').encode(
            x='math_score:Q',
        )

        swedish = base.mark_point(color='red').encode(
            x='swedish_score:Q',
        )

        text = base.mark_text(
            align='left',
            baseline='middle',
        ).encode(
            text='average_score:Q'
        )

        st.write((bar + english + math + swedish + text).properties(width=700))
        st.write(commun_df[['school', 'school_commun', 'type',
                            'average_score', 'english_score', 'math_score', 'swedish_score']])
