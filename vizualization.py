import logging
import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import io
import base64
from datetime import datetime
from matplotlib.figure import Figure
import json
from PIL import Image
from yolov7.yolov7_wrapper import YOLOv7Wrapper

@st.cache_data


def get_data():
    
    
    try:
        results_file = os.path.join(os.path.dirname(__file__), "results.xlsx")
        df = pd.read_excel(results_file)
        
        # Modify the column names to match the new format
        df.columns = ['timestamp', 'model', 'class_id', 'count', 'coordinates', 'confidence']
        
        # Convert the 'timestamp' column to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        st.session_state['df'] = df
        
        return df
    except Exception as e:
        st.error(f"Error loading saved results: {e}")
        return pd.DataFrame()

def get_plot_as_image(fig: Figure):
    """Converts a matplotlib Figure to an image bytes (PNG format)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf.getvalue()



def get_logged_in_user_id():
    """
    Check if the user's ID is already in the session state.
    """
    
    # If the user's details are in the session state, return the user ID.
    if "user" in st.session_state and "uid" in st.session_state["user"]:
        return st.session_state["user"]["uid"]

    # If not found, return None
    st.warning("User not logged in or user ID not available.")
    return None
  
def to_excel(df):
    if df is None or df.empty:
        return b''

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        
    output.seek(0)  # Go to the beginning of the stream
    return output.getvalue()






def parse_coordinates(coord_str):
    # Initialize a default dictionary for coordinates with keys
    default_coord = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
    # Check if 'coord_str' is a string and not NaN or any other type
    if isinstance(coord_str, str):
        # Split the string by comma and convert to integers if possible
        try:
            parts = [int(part) for part in coord_str.split(',')]
            return {'x1': parts[0], 'y1': parts[1], 'x2': parts[2], 'y2': parts[3]}
        except (ValueError, IndexError):
            # Return the default dictionary if conversion fails or not enough parts
            return default_coord
    else:
        # Return the default dictionary if coord_str is not a string
        return default_coord


def preprocess_data(data):
   

    # Initialize a list to store the flattened data
    flattened_data = []

    # Check if the data is empty or contains only non-dictionary entries
    if not any(isinstance(entry, dict) for entry in data.get("all_results", {}).values()):
        st.info("Input data is empty or contains only non-dictionary entries.")
        return pd.DataFrame()

    # Loop through each entry in the data
    for entry in data.get("all_results", {}).values():
        if isinstance(entry, dict):
            # Extract the timestamp and convert it to a readable date, if it exists
            timestamp = entry.get('timestamp', None)
            readable_date = None
            if timestamp:
                try:
                    readable_date = datetime.fromisoformat(timestamp)
                except (ValueError, TypeError):
                    # Handle the case where the timestamp is not in the expected format
                    st.warning(f"Invalid timestamp in entry: {timestamp}")
                    readable_date = None

            # Extract other details from the entry
            model = entry.get('model', None)
            inference_details = entry.get('inference_details', {})
            
            # Check if inference_details is a dictionary and proceed
            if isinstance(inference_details, dict):
                # Add the timestamp and model to the details
                inference_details['timestamp'] = readable_date
                inference_details['model'] = model
                
                # Extract and clean 'count' data
                count = inference_details.get('count', 1)
                inference_details['count'] = parse_count(count)
                
                # Append the cleaned details to the flattened_data list
                flattened_data.append(inference_details)
        else:
            st.warning(f"Unexpected data format: {entry}")

    # Convert the flattened data into a pandas DataFrame
    df = pd.DataFrame(flattened_data)
    
    # Convert the 'timestamp' column to datetime objects, if it exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Handle the case where 'class_id' is not present in the data
    if 'class_id' not in df.columns:
        df['class_id'] = 'Unknown'
    
    st.info(f"Preprocessed data has {len(df)} rows.")
    return df


def parse_count(count):
    # Ensure count is an integer, even when it's a list or in an unexpected format
    if isinstance(count, list) and count and isinstance(count[0], (int, float)):
        return int(count[0])
    elif isinstance(count, (int, float)):
        return int(count)
    return 1  # Default to 1 if count is missing or in an unexpected format


def generate_visualizations(df):
      # Check if the DataFrame is empty
      if df.empty:
            st.write("No data to visualize.")
            return

      # Create a two-column layout
      col1, col2 = st.columns(2)
      
      # Visualization 1: Bar Chart of Object Frequencies
      with col1:
            st.subheader('Object Frequencies')
            fig, ax = plt.subplots()
            df['class_id'].value_counts().head(10).plot(kind='bar',  ax=ax)
            ax.set_ylabel('Frequency')
            ax.set_title('Top 10 Detected Objects')
            st.pyplot(fig)

      # Visualization 2: Line Chart of Detections Over Time
      with col2:
        st.subheader('Detections Over Time')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Filter the DataFrame to only include confidence scores >= 0.1
        high_confidence_df = df[df['confidence'] >= 0.1]
        
        # Group the data by day and sum the counts
        high_confidence_df['timestamp'] = pd.to_datetime(high_confidence_df['timestamp'])
        high_confidence_df.set_index('timestamp').groupby(pd.Grouper(freq='D'))['count'].sum().plot(ax=ax)
        
        ax.set_ylabel('Number of Detections')
        ax.set_title('Daily Detections Trend')
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        st.pyplot(fig)

          # Visualization: Confidence Score Distribution with distinct colors for each class_id
      with col1:
        st.subheader('Confidence Score Distribution by Class ID and Model')
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        # Extract unique class IDs and colors from the palette
        class_ids = df['class_id'].unique()
        models = df['model'].unique()
        colors = sns.color_palette('Set2', n_colors=len(class_ids) * len(models))
        
        # Plot histogram for each class ID and model
        for i, class_id in enumerate(class_ids):
            class_df = df[df['class_id'] == class_id]
            bins = np.linspace(0.1, 1.0, 21)  # 20 bins from 0.1 to 1.0
            hist, bin_edges = np.histogram(class_df['confidence'], bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Calculate the count for each model
            model_counts = []
            for j, model in enumerate(models):
                model_df = class_df[class_df['model'] == model]
                model_hist, _ = np.histogram(model_df['confidence'], bins=bins)
                model_counts.append(model_hist)
            
            # Plot the stacked bars
            bottom = np.zeros_like(bin_centers)
            for j, model_count in enumerate(model_counts):
                ax.bar(bin_centers, model_count, width=0.04, color=colors[i * len(models) + j], alpha=0.7, bottom=bottom, label=f"{models[j]}")
                bottom += model_count
                
                # Add count on top of each bar
                for k, c in enumerate(model_count):
                    ax.text(bin_centers[k], bottom[k] - c / 2, str(int(c)), ha='center', va='center', fontsize=8)
        
        ax.set_title('Confidence Scores Histogram by Class ID and Model')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.set_xlim(0.1, 1.0)
        ax.legend(title='Model', ncol=2)
        
        st.pyplot(plt.gcf())
      # Visualization 4: Heatmap of Detections by Model and Class
      with col2:
        df = pd.read_excel('results.xlsx', engine='openpyxl')
        if 'model' in df.columns and 'class_id' in df.columns:
            st.subheader('Heatmap of Detections by Model and Class')
            counts = df.groupby('model')['coordinates'].count()
            # Create a pivot table with the sum of 'count' for each model and class_id
            pivot_table = pd.pivot_table(df, values='coordinates', index='model', columns='class_id', aggfunc='count')
            
            # Fill any missing values with 0
            pivot_table = pivot_table.fillna(0)
            
            
            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot the heatmap
            sns.heatmap(pivot_table, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
            
            # Set the title and axis labels
            ax.set_title('Detections by Model and Class')
            ax.set_xlabel('Class ID')
            ax.set_ylabel('Model')
            
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=90)
            
            # Display the plot
            st.pyplot(fig)



def get_filter_inputs(df, identifier):
    try:
        # Ensure the default values are lists in the session state
        if 'selected_models' not in st.session_state:
            st.session_state['selected_models'] = []
        if 'selected_class_ids' not in st.session_state:
            st.session_state['selected_class_ids'] = []

        # Model filter
        model_options = df['model'].unique().tolist() if 'model' in df.columns else []
        st.sidebar.divider()
        st.sidebar.markdown("### 📇 Select the parameters below to filter the dataset.")

        # Update the session state if default models are not in options
        if not set(st.session_state['selected_models']).issubset(set(model_options)):
            st.session_state['selected_models'] = []

        selected_models = st.sidebar.multiselect(
            'Select Model(s)',
            model_options,
            default=st.session_state['selected_models']
        ) if model_options else []

        st.session_state['selected_models'] = selected_models

        # Class ID filter - dynamically update based on selected models
        class_id_options = df[df['model'].isin(selected_models)]['class_id'].unique().tolist() if 'class_id' in df.columns and selected_models else df['class_id'].unique().tolist() if 'class_id' in df.columns else []

        # Update the session state if default class IDs are not in options
        if not set(st.session_state['selected_class_ids']).issubset(set(class_id_options)):
            st.session_state['selected_class_ids'] = []

        selected_class_ids = st.sidebar.multiselect(
            'Select Class ID(s)',
            class_id_options,
            default=st.session_state['selected_class_ids']
        ) if class_id_options else []

        st.session_state['selected_class_ids'] = selected_class_ids

        # Date filter
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S') if isinstance(x, str) and re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', x) else x)
                date_min = df['timestamp'].min().date()
                date_max = df['timestamp'].max().date()
                selected_date_range = st.sidebar.date_input(
                    'Select Date Range', 
                    [date_min, date_max]
                )
            except (ValueError, TypeError):
                st.sidebar.warning("Invalid timestamp data format. Date filtering will be disabled.")
                selected_date_range = [datetime.now().date(), datetime.now().date()]
        else:
            st.sidebar.warning("No timestamp data available for filtering.")
            selected_date_range = [datetime.now().date(), datetime.now().date()]

        # Return a dictionary of filter options
        filter_options = {
            'selected_models': selected_models,
            'selected_class_ids': selected_class_ids,
            'selected_date_range': selected_date_range
        }

        return filter_options

    except Exception as e:
        st.error(f"Error in get_filter_inputs: {e}")
        # If there's an error, return an empty dictionary or some default values
        return {}




def apply_filters(df,filter_options):
    # Check if 'df' and 'filter_options' are available in the session state
    if 'df' in st.session_state and 'filter_options' in st.session_state:
        df = st.session_state['df']
        filter_options = st.session_state['filter_options']

        # Unpack filter options
        # selected_models = filter_options['selected_models']
        # selected_class_ids = filter_options['selected_class_ids']  # Unpack the selected_class_ids
        # selected_date_range = filter_options['selected_date_range']

        # # Apply filters to the DataFrame
        # # Applying model, class_id, and date range filters
        # filtered_df = df[
        #     df['model'].isin(selected_models) &
        #     df['class_id'].isin(selected_class_ids) &  # Apply class_id filter
        #     df['timestamp'].dt.date.between(*selected_date_range)
        # ]
        
        # # Update the session state with the filtered dataframe
        # st.session_state['filtered_data'] = filtered_df
        # st.session_state['filtered'] = True

         # Unpack filter options
        selected_models = filter_options['selected_models']
        selected_class_ids = filter_options['selected_class_ids']
        selected_date_range = filter_options['selected_date_range']

        # Apply filters to the DataFrame
        filtered_df = df[
            df['model'].isin(selected_models) &
            df['class_id'].isin(selected_class_ids) &
            df['timestamp'].dt.date.between(*selected_date_range)
        ]
        
        return filtered_df

        # Provide feedback about the operation
        if filtered_df.empty:
            st.sidebar.warning("No data matches the filters.")
        else:
            st.sidebar.success(f"Filtered data contains {len(filtered_df)} rows.")

        # No need to rerun the page unless there is a specific reason to do so
        # st.experimental_rerun()

    else:
        st.sidebar.error("Data or filter options are not set in the session state.")





def update_filtered_data():
    # Check if the dataframe is available in the session state
    if 'df' in st.session_state and 'class_id' in st.session_state['df']:
        # Get the current selection from session state
        selected_value = st.session_state.class_id_select
        # Update the filtered dataframe in the session state
        st.session_state.filtered_data = st.session_state['df'][st.session_state['df']['class_id'] == selected_value]



    

def visualize_inferences():


    st.session_state['filtered'] = False
    
    # Load the saved results
    df = get_data()

    if df is None or df.empty:
        st.markdown("## 📊 Visualizations & Insights")
        st.markdown("### 🙈 Oops!")
        st.write("It seems there's an issue with your data or you don't have any data uploaded yet.")
        st.write("Upload your data and start seeing insights")
        return

    st.header('📊 :green[Visualizations & Insights]')
    st.divider()
    
    # Create an expander for Summary Statistics
    with st.expander("**Summary Statistics** ", expanded=False):
        st.subheader(':blue[Summary Statistics] 📈')
        st.markdown(f"**Total inferences:** {len(df)}")
        st.markdown(f"**Unique objects detected:** {df['class_id'].nunique()}")
        st.markdown(f"**Average count of detections:** {df['count'].mean():.2f}")

        # Display Bar chart for object frequencies and Pie chart for proportion of detected objects side-by-side
        col1, col2 = st.columns(2)

        with col1:
            # Count the occurrences of each class_id and sort them in descending order
            class_id_counts = df["class_id"].value_counts()

            # Select the top 12 most frequent class_ids
            top_12_class_ids = class_id_counts.head(12)
            st.write(':blue[**📊 Frequency of Detected Objects**]')
            fig1, ax1 = plt.subplots()
            top_12_class_ids.plot(kind="bar", ax=ax1)
            st.pyplot(fig1,use_container_width=True)

        # Visualization 2: Pie Chart of Detected Objects Proportion by Model
        with col2:
            st.write('🍩 :blue[**Proportion of Detected Objects by Model**]')
            
            # Get the unique models
            models = df['model'].unique()
            
            # Calculate the total count of detected objects
            total_count = df['count'].sum()
            
            # Create a figure with a single pie chart
            fig, ax = plt.subplots(figsize=(3, 3))
            
            # Calculate the proportion of detected objects for each model
            model_proportions = []
            model_labels = []
            for model in models:
                model_df = df[df['model'] == model]
                model_count = model_df['count'].sum()
                model_proportion = model_count / total_count
                model_proportions.append(model_proportion)
                model_labels.append(model)
            
            # Plot the pie chart
            ax.pie(model_proportions, labels=model_labels, autopct="%1.1f%%")
            ax.set_title("Proportion of Detected Objects by Model")
            
            st.pyplot(fig)
    
    # Layout for data table and filter options
    col1 = st.columns(1)[0]

    with col1:
        with st.expander("Data", expanded=True):
            st.subheader("📜 :blue[**Data table**]")
            st.dataframe(st.session_state.get('filtered_data', df), use_container_width=True)
            
            st.download_button(
                label="Download Data as Excel",
                data=to_excel(st.session_state.get('filtered_data', df)),
                file_name='full_data.xlsx',
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # Visualization toggle
    show_saved_viz = st.checkbox('Show Visualizations for Saved Results')
    if show_saved_viz:
        generate_visualizations(df)

    #---------------Time Series Visualization------------------------------------#                 
    with st.expander("**Historical Detection Analysis**", expanded=False):
        st.subheader(':blue[Detection Trends Over Time] 📅')
        col1, col2 = st.columns(2)
        
        # Convert the timestamp column to datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Specify colors and font
        colors = ['#E63946', '#1D3557', '#457B9D']
        font = {'family': 'serif',
                'color':  'green',
                'weight': 'normal',
                'size': 12,
                }

        with col1:
            st.write(':green[**Total Detections Over Time by Model**]')
            fig, ax = plt.subplots(figsize=(10, 5))
            for model in df['model'].unique():
                df_model = df[df['model'] == model]
                df_model_grouped = df_model.groupby(pd.Grouper(key='timestamp', freq='D'))['count'].sum().reset_index()
                df_model_grouped.plot(kind='line', x='timestamp', y='count', ax=ax, label=model)
            ax.legend()
            ax.set_ylabel("Number of Detections", fontdict=font)
            ax.set_xlabel("Date", fontdict=font)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

        if 'class_id' in df.columns:
            with col2:
                st.write(':green[**Unique Objects Detected Over Time**]')
                fig, ax = plt.subplots(figsize=(10, 5))
                df_unique_objects = df.groupby(pd.Grouper(key='timestamp', freq='D'))['class_id'].nunique().reset_index()
                df_unique_objects.plot(kind='line', x='timestamp', y='class_id', color=colors[1], ax=ax)
                ax.set_ylabel("Number of Unique Objects", fontdict=font)
                ax.set_xlabel("Date", fontdict=font)
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

        with col1:
            st.write(':green[**Model Utilization Over Time**]')
            fig, ax = plt.subplots(figsize=(10, 5))
            df_model_usage = df.groupby(['timestamp', 'model'])['count'].sum().unstack().fillna(0)
            df_model_usage.plot(kind='line', ax=ax)
            ax.set_ylabel("Usage Frequency", fontdict=font)
            ax.set_xlabel("Date", fontdict=font)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)




      
      






            





