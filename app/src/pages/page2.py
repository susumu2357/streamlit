import streamlit as st
from ..utils import Page


class Page2(Page):
    def __init__(self, state):
        self.state = state

    def write(self):
        st.title("Page 2")

        st.write(self.state.client_config["slider_value"])

        left_column, right_column = st.beta_columns(2)
        pressed = left_column.button('Press me?')
        if pressed:
            right_column.write("Woohoo!")

        expander = st.beta_expander("FAQ")
        expander.write("Here you could put in some really, really long explanations...")