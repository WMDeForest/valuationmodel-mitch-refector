@st.fragment
def component_handler(df):

    discount_rate = (
        st.number_input(
            "Discount Rate (%)",
            key="text_input_value",
            min_value=0.00,
            max_value=10.00,
            value=4.50,
            step=0.01,
            format="%.2f",
        )
        / 100
    )

    songs = sorted(df["Track"].unique(), key=lambda x: x.lower())
    selected_songs = st.multiselect(
        "Select Songs", songs, default=songs, key="selected_songs"
    )

    if discount_rate and selected_songs:
        calculate_graph(df, discount_rate=discount_rate, selected_songs=selected_songs)
    else:
        if st.button("Run All"):
            calculate_graph(
                df=df,
                discount_rate=st.session_state.text_input_value,
                selected_songs=st.session_state.selected_songs,
            )