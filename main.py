import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import map_week_category, add_day_of_week, calculate_perc_columns, find_matching_rows, find_matching_column, day_mapping
from statistics import mode
from io import StringIO

base="white"

st.set_page_config(
    layout="wide",
    page_title="Matches Coffees",
    page_icon="☕︎",
) 

def run():
        """
        Sets up the dashboard and the side bar for all pages
        """

        ################################################################################
        ###                                                                          ###
        ###                         SET UP SIDEBAR                                   ###
        ###                                                                          ###
        ################################################################################

        with st.sidebar:   
            st.title("Matched Coffees")  
            markdown_text = """
            From my experience analysing my own fitness activities (pace, routes, dates...), I was wondering if similar functionality could be applied to my coffee consumption habits to find patterns and insights. So this app aims to replicate the popular fitness app Strava’s [Matched Activities](https://support.strava.com/hc/en-us/articles/216918597-Matched-Activities/) functionality. When you process your coffee data, the app will search for similar consumptions, cluster them together, and allow you to select the group you want to visualise. Developed by [Alvaro Ager](https://www.linkedin.com/in/alvaroager/)
            """
            st.markdown(markdown_text)



        ################################################################################
        ###                                                                          ###
        ###                             INTRODUCTION                                 ###
        ###                                                                          ###
        ################################################################################
    
        # Displat hero image
        st.image('hero.png', use_column_width=True)

        st.title("")
        st.title("Introduction")

        # Display markdown text
        markdown_text = """
        From my experience analysing my own fitness activities (pace, routes, dates...), I was wondering if similar functionality could be applied to my coffee consumption habits to find patterns and insights. So this app aims to replicate the popular fitness app Strava’s [Matched Activities](https://support.strava.com/hc/en-us/articles/216918597-Matched-Activities/) functionality. When you process your coffee data, the app will search for similar consumptions, cluster them together, and allow you to select the group you want to visualise. Developed by [Alvaro Ager](https://www.linkedin.com/in/alvaroager/)

        This is how the feature looks in Strava:
        """
        st.markdown(markdown_text)

        # Display markdown text
        # st.markdown("When you process your statement, the app will search for similar transactions, cluster them together, and allow you to select the group you want to visualise.")


        # Display the image
        st.image('https://support.strava.com/hc/article_attachments/4413210049933', use_column_width=True)
        
        # Display horizontal line
        st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)

        ################################################################################
        ###                                                                          ###
        ###                             APPLICATION                                  ###
        ###                                                                          ###
        ################################################################################

        # Streamlit app
        st.title("Load the data and play around! 📖")
        st.subheader("Select a pre-loaded dataset or upload a file")
        st.markdown("Click 'Upload CSV File' 👇 and it will display the instructions")

        # Set up a sample dataset
        data_str = """Date,CoffeeShop,Amount,Amount of Products
        25/08/2024,Rocket Bean Roastery Miera,-15.23,5
        01/09/2024,Rocket Bean Roastery Miera,-19.12,4
        01/09/2024,Rocket Bean Roastery Miera,-2.5,1
        08/09/2024,Rocket Bean Roastery Miera,-17.45,7
        15/09/2024,Rocket Bean Roastery Miera,-16.67,3
        22/09/2024,Rocket Bean Roastery Miera,-14.78,6
        22/09/2024,Rocket Bean Roastery Miera,-2.5,1
        29/09/2024,Rocket Bean Roastery Miera,-18.34,8
        29/09/2024,Rocket Bean Roastery Miera,-2.5,1
        07/10/2024,Rocket Bean Roastery Miera,-15.89,5
        12/10/2024,Kukul Ceptuve,-13.99,6
        13/10/2024,Kukul Ceptuve,-22.45,5
        20/10/2024,Kukul Ceptuve,-18.34,8
        26/10/2024,Kukul Ceptuve,-21.67,4
        27/10/2024,Kukul Ceptuve,-15.56,9
        02/11/2024,Kukul Ceptuve,-19.89,6
        03/11/2024,Kukul Ceptuve,-24.23,7
        21/08/2024,Rocket Bean Brivibas,-5.45,2
        22/08/2024,Rocket Bean Brivibas,-8.67,1
        23/08/2024,Rocket Bean Brivibas,-4.23,1
        24/08/2024,Rocket Bean Brivibas,-6.12,2
        25/08/2024,Rocket Bean Brivibas,-7.99,2
        26/08/2024,Rocket Bean Brivibas,-5.45,1
        27/08/2024,Rocket Bean Brivibas,-9.99,1
        28/08/2024,Rocket Bean Brivibas,-10.67,2
        29/08/2024,Rocket Bean Brivibas,-4.34,2
        30/08/2024,Rocket Bean Brivibas,-5.89,1
        04/09/2024,Kalve Coffee Roasters,-3.34,1
        05/09/2024,Kalve Coffee Roasters,-5.78,2
        06/09/2024,Kalve Coffee Roasters,-4.89,1
        07/09/2024,Kalve Coffee Roasters,-7.45,2
        08/09/2024,Kalve Coffee Roasters,-6.23,1
        09/09/2024,Kalve Coffee Roasters,-3.78,2
        10/09/2024,Kalve Coffee Roasters,-8.45,1
        11/09/2024,Kalve Coffee Roasters,-4.67,2
        12/09/2024,Kalve Coffee Roasters,-6.99,1
        13/09/2024,Kalve Coffee Roasters,-7.12,1
        14/09/2024,Kalve Coffee Roasters,-4.45,2
        15/09/2024,Kalve Coffee Roasters,-6.78,1
        16/09/2024,Kalve Coffee Roasters,-3.56,1
        21/09/2024,Mr. Biskvits,-12.45,5
        22/09/2024,Mr. Biskvits,-9.34,4
        28/09/2024,Mr. Biskvits,-15.67,6
        29/09/2024,Mr. Biskvits,-18.12,5
        05/10/2024,Mr. Biskvits,-19.78,4
        06/10/2024,Mr. Biskvits,-8.34,6
        12/10/2024,Mr. Biskvits,-17.56,5
        13/10/2024,Mr. Biskvits,-20.45,4
        04/08/2024,Kure Ceptuve,-4.34,1
        11/08/2024,Kure Ceptuve,-6.78,2
        18/08/2024,Kure Ceptuve,-8.45,1
        25/08/2024,Kure Ceptuve,-3.67,1
        01/09/2024,Kure Ceptuve,-7.12,2
        08/09/2024,Kure Ceptuve,-5.56,1
        15/09/2024,Kure Ceptuve,-6.89,1
        22/09/2024,Kure Ceptuve,-3.45,2
        29/09/2024,Kure Ceptuve,-9.78,1
        06/10/2024,Kure Ceptuve,-4.23,2
        30/09/2024,Kukotava,-3.34,1
        08/10/2024,Kukotava,-4.12,1
        15/10/2024,Kukotava,-3.45,1
        """

        # Use StringIO to convert the string to a file-like object
        data_file = StringIO(data_str)

        # Read the data into a pandas DataFrame
        df1 = pd.read_csv(data_file, parse_dates=['Date'], dayfirst=True).sort_values('Date', ascending = False)

        # List of available DataFrames
        dataframes = {'DataFrame 1': df1}

        ################################################################################
        ###                                                                          ###
        ###               CREATE FILTER TO SELECT DATAFRAME OR CSV UPLOAD            ###
        ###                                                                          ###
        ################################################################################

        # Option to select a pre-loaded DataFrame
        selected_option = st.radio("Select Option:", ["Select Pre-loaded DataFrame", "Upload CSV File"])

        if selected_option == "Select Pre-loaded DataFrame":
            # Allow the user to select a DataFrame
            df = df1

        elif selected_option == "Upload CSV File":

        ################################################################################
        ###                                                                          ###
        ###                       INSTRUCTIONS  IF CSV FILE                          ###
        ###                                                                          ###
        ################################################################################
            # Display title 
            st.header('Instructions')

            # Display markdown text
            st.markdown("You need to select or drag and drop a CSV file with your data on the next section. Template to follow: ")
            
            # Create dataframe example to showcase if user wants to create a custom dataframe to upload
            example_data = {
                'Date': ['2019-08-16 06:59:37', '2019-08-27 06:34:04', '2019-08-28 16:34:19'],
                'CoffeeShop': ['Starbucks', 'Starbucks', 'Costa Coffee'],
                'Amount': [15, -2, -3.54],
            }
            example_data = pd.DataFrame(example_data)

            # Display dataframe in the Streamlit interface
            st.table(example_data)

            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

            if uploaded_file is not None:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(uploaded_file)
    

        # Display the selected data
        if 'df' in locals():
            st.write("This is my input data:")
            st.write(df)

            # Display horizontal line
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
            st.title("Results: Matched Coffees ☕︎") 


        ################################################################################
        ###                                                                          ###
        ###                     APPLY FUNCTIONS AND MATCH THE ROWS                   ###
        ###                                                                          ###
        ################################################################################

            # Convert the "Date" column to datetime
            df["Date"] = pd.to_datetime(df["Date"])
            df["Amount"] = df["Amount"].astype(float).round(2)

            # Data preprocessing
            add_day_of_week(df)
            calculate_perc_columns(df)

            # Add the day of the week to the dataframe
            df['day_of_week'] = df['day_of_week'].map(day_mapping)

            # Find matching rows and create 'matched_rows' column
            df['matched_rows'] = df.apply(lambda row: find_matching_rows(row, df), axis=1)
            df['matched_rows'] = df['matched_rows'].apply(lambda x: sorted(x) if isinstance(x, list) else [])

            required_columns = ['Amount', 'Date', 'CoffeeShop']



        ################################################################################
        ###                                                                          ###
        ###               CREATE LOGIC FOR THE MERCHANT AND ROW FILTERS              ###
        ###                                                                          ###
        ################################################################################

            if all(col in df.columns for col in required_columns) and df.shape[0] > 0:

                # Filter merchants with non-empty 'matched_rows'
                filtered_merchants = df[df['matched_rows'].apply(lambda x: len(x) > 0)]['CoffeeShop'].dropna().unique()
                if len(filtered_merchants) == 0:
                    st.warning("No merchants with matching rows found.")
                    return

                selected_merchant = st.selectbox("Select a coffee shop:", filtered_merchants)

                # Filter rows based on selected merchant
                pre_filtered_df = df[df['CoffeeShop'] == selected_merchant]

                # Filter rows with non-empty 'matched_rows'
                pre_filtered_df = pre_filtered_df = pre_filtered_df[pre_filtered_df['matched_rows'].apply(lambda x: isinstance(x, list) and len(x) >= 0)].reset_index()

                # pre_filtered_df = pre_filtered_df.reset_index()

                if pre_filtered_df.shape[0] == 0:
                    st.warning(f"No matching rows found for {selected_merchant}.")
                    return

                # Create a selection dropdown for rows
                selected_row = st.selectbox("Select a purchase number:", pre_filtered_df[pre_filtered_df['matched_rows'].apply(lambda x: len(x) > 0)].dropna()['index'].unique())

                selected_row_indices = pre_filtered_df[pre_filtered_df['index'] == selected_row].matched_rows.iloc[0]

                filtered_df = pre_filtered_df[pre_filtered_df['index'].isin(selected_row_indices)]



        ################################################################################
        ###                                                                          ###
        ###               CALCULATE INSIGHTS ON THE FLY BASED ON FILTERS             ###
        ###                                                                          ###
        ################################################################################

                # Find matching names and create 'matched_names' column
                filtered_df['matched_names'] = filtered_df.apply(lambda row: find_matching_column(row, filtered_df, 'CoffeeShop'), axis=1)
                filtered_df['mode_matched_names'] = filtered_df['matched_names'].apply(lambda x: mode(x) if len(x) > 0 else None)

                # Find matching days of the week and create 'matched_days_week' column
                filtered_df['matched_days_week'] = filtered_df.apply(lambda row: find_matching_column(row, filtered_df, 'week_category'), axis=1)
                filtered_df['mode_matched_days_week'] = filtered_df['matched_days_week'].apply(lambda x: mode(x) if len(x) > 0 else None)


                # Calculate time differences between consecutive dates
                filtered_df["Time Difference"] = filtered_df["Date"].diff()

                # Calculate the average time difference
                average_time_diff = filtered_df["Time Difference"].median()

                # Display horizontal line
                st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)

                # Display section with KPIs
                col1, col2, col3, col4 = st.container().columns(4)
                with col1:
                    st.metric(label = "Similar coffee purchases in this group", value = filtered_df['Amount'].count())
                with col2:
                    st.metric(label = "Average amount", value = round(filtered_df['Amount'].mean(), 2))
                with col3:
                    st.metric(label = "When usually happen", value = filtered_df['mode_matched_days_week'].iloc[0])
                with col4:
                    st.metric(label = "Usual days between coffees", value = average_time_diff.days * (-1))
                
                # Display horizontal line   
                st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)



        ################################################################################
        ###                                                                          ###
        ###                      VISUALISATION: CHART AND TABLE                      ###
        ###                                                                          ###
        ################################################################################

                # Scatter plot
                fig = go.Figure()

                # Scatter plot for 'Amount'
                fig.add_trace(go.Scatter(
                    x=filtered_df['Date'],
                    y=filtered_df['Amount'],
                    mode='markers',
                    text=filtered_df['CoffeeShop'],
                    marker=dict(
                        size=10,
                        color='darkorange',
                        opacity=0.7
                    ),
                    name='Amount'
                ))

                # Line trace for the average in orange
                fig.add_trace(go.Scatter(
                    x=filtered_df['Date'],
                    y=[filtered_df['Amount'].mean()] * len(filtered_df),  # Same average value for all points
                    mode='lines',
                    line=dict(color='lightblue', width=2),
                    name='Average'
                ))

                fig.update_layout(
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Amount')
                )

                # Display the plot in Streamlit
                st.plotly_chart(fig, use_container_width=True)

                # Display the filtered DataFrame
                st.table(filtered_df[['CoffeeShop', 'Date', 'Amount', 'day_of_week']].sort_values(by='Date'))

            # Show warning messages if the input dataframe is incorrect
            else:
                missing_columns = [col for col in required_columns if col not in df.columns]
                if df.shape[0] == 0:
                    st.warning("The DataFrame has no rows of data (excluding headers).")
                elif missing_columns:
                    st.warning(f"The following columns are missing: {', '.join(missing_columns)}")


        # Display horizontal line
        st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)
        st.title("Analysing results 🧪") 
        st.markdown("""
        By selecting coffee shops like 'Mr. Biskvits', data shows I go there during the weekends and buy different products, given the amount spent. That makes sense after an excursion around Sigulda! 😜 On other coffee shops like Rocket Bean Brivibas, I usually grab a coffee to go on my way to the office.

        Moreover, you can see I like drinking an additional coffee after weekend's brunch - coz why not? There's more time to chill out.          

        What can you find amongst your data? If you're visiting Latvia and you're a coffee addict these places are a must!
        """)
run()   