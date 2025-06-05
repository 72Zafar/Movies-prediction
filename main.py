import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import  pickle

n = 10000

df = pd.DataFrame({

    'MovieID': np.arange(n),

    'Title': [f'Movie {i}' for i in range(n)],

    'Genre': np.random.choice(['Drama', 'Comedy', 'Action', 'Horror'], size=n),

    'ReleaseYear': np.random.randint(1980, 2024, size=n),

    'Rating': np.round(np.random.normal(loc=6.5, scale=1.5, size=n), 1),

    'Votes': np.random.randint(100, 100000, size=n),

    'RevenueMillions': np.round(np.random.uniform(1, 300, size=n), 2)

})



try:
    conn = sqlite3.connect('movies.db')
    df.to_sql('movies', conn, if_exists='replace', index=False)
    conn.close()
except Exception as e:
    st.error(f"Failed to write to SQLite: {e}")







# Load data
conn = sqlite3.connect('movies.db')
df = pd.read_sql('SELECT * FROM movies', conn)
df['Genre'] = df['Genre'].astype(str)  # Ensure Genre is string type

# title
st.markdown(
    "<h1 style='color: #1f77b4;'>Movie Data Analysis</h1>",
    unsafe_allow_html=True
)




option = st.sidebar.radio("Choose a Section", ["EDA", "ML"])

if option == "EDA":
    st.subheader("Exploratory Data Analysis")
    st.write("Data Overview")

    # Show the first few rows of the dataframe
    st.dataframe(df.head())

    # Show basic statistics
    st.write("### Basic Statistics")
    st.write(df.describe())

    # Show unique genres
    st.write("### Unique Genres")
    st.write(df['Genre'].unique())
    
    # Filters
    genre_filter = st.selectbox("Select a genre", df['Genre'].unique())
    year_range = st.slider("Select Release Year Range", 1980, 2023, (2000, 2020))

    # # Filtered data
    filtered_df = df[
            (df['Genre'] == genre_filter) &
            (df['ReleaseYear'].between(*year_range))
    ]

    #  Show metrics
    st.metric("Average Rating", round(filtered_df['Rating'].mean(), 2))
    st.metric("Average Revenue", f"${round(filtered_df['RevenueMillions'].mean(), 2)}M")
    
    st.title("Revenue and Ratings Analysis")

    fig, ax = plt.subplots(figsize=(15, 6))
    avg_revenue = filtered_df.groupby('ReleaseYear')['RevenueMillions'].mean()
    ax.plot(avg_revenue.index, avg_revenue.values, color='red' ,label=f"{genre_filter}")
    ax.set_title("Average Revenue by Release Year")
    ax.legend(loc='upper left')
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Average Revenue (Millions)")
    st.pyplot(fig)

   # Plotting Ratings and Votes by Release Year
    st.title("Ratings by Release Year")
    fig, ax = plt.subplots(figsize=(15, 6))
    avg_rating = filtered_df.groupby('ReleaseYear')['Rating'].mean()
    ax.plot(avg_rating.index,avg_rating.values, alpha=0.9, color='red' ,label=f"{genre_filter}")
    ax.legend(loc='upper left')
    ax.set_title("Ratings by Release Year")
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Rating")
    st.pyplot(fig)

   # Plotting Votes by Release Year
    st.title("Votes by Release Year")
    fig, ax = plt.subplots(figsize=(15, 6))
    avg_votes = filtered_df.groupby('ReleaseYear')['Votes'].mean()
    ax.plot(avg_votes.index, avg_votes.values, alpha=0.9, color='red',label=f"{genre_filter}" )
    ax.legend(loc='upper left')
    ax.set_title("Votes by Release Year")
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Votes")
    st.pyplot(fig)

else:
    option = "ML"
    st.subheader("Machine Learing Section")

    # import ML models
    pipe = pickle.load(open('movie_model.pkl', 'rb'))
    # Load the data
    df =  pickle.load(open('movie_data.pkl', 'rb'))

    st.title("Movie RevenueMillions Prediction")
 
    #  title 
    Move_Title = st.selectbox("Move Titel", df['Title'].unique())

    # Genre
    Genre = st.selectbox("Genre", df['Genre'].unique())
    
    # Release Year
    Release_Year = st.selectbox("Release Year", df['ReleaseYear'].unique())

    # Rating
    Rating = st.number_input("Rating", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    if Rating < 0.0 or Rating > 10.0:
        st.error("Rating must be between 0.0 and 10.0")
    
    # Votes
    Votes = st.number_input("Votes", min_value=0, max_value=1000000, value=50000, step=1000)

    # Move Score
    Move_Score = st.number_input("Move Score", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    
    # Predict button
    if st.button("Predict RevenueMillions"):
        # Prepare the input data
        input_data = pd.DataFrame({
            'Title': [Move_Title],
            'Genre': [Genre],
            'ReleaseYear': [Release_Year],
            'Rating': [Rating],
            'Votes': [Votes],
            'MovieScore': [Move_Score]
        })
        # Make prediction
        prediction = pipe.predict(input_data)
        st.success(f"Predicted RevenueMillions: ${prediction[0]:.2f}M")


