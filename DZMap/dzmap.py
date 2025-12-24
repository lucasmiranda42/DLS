import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import copy
import io
import os
import base64

from sys import argv


# Set page configuration with light theme
st.set_page_config(
    page_title="DZMap - DLS Score Analysis", page_icon="📊", layout="wide"
)

# Set up color palette - ensure consistent colors across all plots
palette = sns.color_palette("Set2", as_cmap=True)
color_discrete_sequence = sns.color_palette("Set2").as_hex()

# Define consistent color mapping
COLOR_MAP = {
    "Control": color_discrete_sequence[0],  # First color in Set2 for Controls (green)
    "CSDS": color_discrete_sequence[1],      # Second color in Set2 for CSDS (orange)
}

# Set default file path
DEFAULT_FILE = "../sample_data/DLS_domain_scores.csv"


def create_download_button(fig, filename, button_text="Download as SVG", key=None):
    """Create a download button for Plotly figures in SVG format"""
    try:
        # Convert figure to SVG
        svg_bytes = fig.to_image(format="svg")
        
        # Create download button
        st.download_button(
            label=button_text,
            data=svg_bytes,
            file_name=f"{filename}.svg",
            mime="image/svg+xml",
            key=key
        )
    except Exception as e:
        st.error(f"Error creating download button: {e}")


def create_download_buttons_row(fig, filename_base, key_base):
    """Create a row of download buttons for different formats"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            svg_bytes = fig.to_image(format="svg")
            st.download_button(
                label="📊 Download SVG",
                data=svg_bytes,
                file_name=f"{filename_base}.svg",
                mime="image/svg+xml",
                key=f"{key_base}_svg"
            )
        except Exception as e:
            st.error(f"SVG export error: {e}")
    
    with col2:
        try:
            pdf_bytes = fig.to_image(format="pdf")
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes,
                file_name=f"{filename_base}.pdf",
                mime="application/pdf",
                key=f"{key_base}_pdf"
            )
        except Exception as e:
            st.error(f"PDF export error: {e}")
    
    with col3:
        try:
            png_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
            st.download_button(
                label="🖼️ Download PNG",
                data=png_bytes,
                file_name=f"{filename_base}.png",
                mime="image/png",
                key=f"{key_base}_png"
            )
        except Exception as e:
            st.error(f"PNG export error: {e}")


def create_download_buttons_row_with_csv(fig, filename_base, key_base, data_df, data_filename=None):
    """Create a row of download buttons for different formats including CSV data"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            svg_bytes = fig.to_image(format="svg")
            st.download_button(
                label="📊 Download SVG",
                data=svg_bytes,
                file_name=f"{filename_base}.svg",
                mime="image/svg+xml",
                key=f"{key_base}_svg"
            )
        except Exception as e:
            st.error(f"SVG export error: {e}")
    
    with col2:
        try:
            pdf_bytes = fig.to_image(format="pdf")
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes,
                file_name=f"{filename_base}.pdf",
                mime="application/pdf",
                key=f"{key_base}_pdf"
            )
        except Exception as e:
            st.error(f"PDF export error: {e}")
    
    with col3:
        try:
            png_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
            st.download_button(
                label="🖼️ Download PNG",
                data=png_bytes,
                file_name=f"{filename_base}.png",
                mime="image/png",
                key=f"{key_base}_png"
            )
        except Exception as e:
            st.error(f"PNG export error: {e}")
    
    with col4:
        try:
            csv_data = data_df.to_csv(index=False)
            csv_filename = data_filename if data_filename else f"{filename_base}_data.csv"
            st.download_button(
                label="📋 Download CSV",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv",
                key=f"{key_base}_csv"
            )
        except Exception as e:
            st.error(f"CSV export error: {e}")


def load_data(file_path=None):
    """Load data from a file or use uploaded file"""
    try:
        # Check if default file exists
        if file_path is None and os.path.exists(DEFAULT_FILE):
            scores_per_domain = pd.read_csv(DEFAULT_FILE, index_col=0)
            st.sidebar.success(f"Loaded default dataset: {DEFAULT_FILE}")
            return process_data(scores_per_domain)

        # Command line argument
        if len(argv) > 1:
            scores_per_domain = pd.read_csv(argv[1], index_col=0)
            return process_data(scores_per_domain)

        # User uploaded file
        if file_path:
            scores_per_domain = pd.read_csv(file_path, index_col=0)
            return process_data(scores_per_domain)

        # No data available
        return None

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def process_data(df):
    """Process dataframe into the format needed for visualization"""
    # Process data similarly to the original script
    scores_per_domain = df.melt(
        id_vars=["Sample name", "Condition"],
        var_name="domain",
        value_name="Z-score value",
    ).sort_values(by=["Condition", "domain", "Z-score value"])
    
    # Replace DLS with CSDS in condition names for display
    scores_per_domain["Condition"] = scores_per_domain["Condition"].replace("DLS", "CSDS")

    return scores_per_domain


def get_original_dataframe(melted_data):
    """Reconstruct the original dataframe format from the melted data"""
    if melted_data is not None:
        original_df = melted_data.pivot_table(
            index=["Sample name", "Condition"], columns="domain", values="Z-score value"
        ).reset_index()
        return original_df
    return None


def calculate_effect_sizes(data):
    """Calculate effect sizes (difference in means) for each domain to sort violin plot"""
    effect_sizes = []
    domains = data['domain'].unique()
    
    for domain in domains:
        domain_data = data[data['domain'] == domain]
        control_mean = domain_data[domain_data['Condition'] == 'Control']['Z-score value'].mean()
        csds_mean = domain_data[domain_data['Condition'] == 'CSDS']['Z-score value'].mean()
        effect_size = abs(csds_mean - control_mean)  # Absolute difference
        effect_sizes.append({'domain': domain, 'effect_size': effect_size})
    
    effect_df = pd.DataFrame(effect_sizes)
    return effect_df.sort_values('effect_size', ascending=False)['domain'].tolist()


def create_violin_plot(data, color_map=None):
    if color_map is None:
        color_map = COLOR_MAP
        
    # Sort domains by effect size
    sorted_domains = calculate_effect_sizes(data)
    
    violin_fig = px.strip(
        data,
        y="Z-score value",
        x="domain",
        color="Condition",
        hover_data={"Sample name": True},
        custom_data=["Sample name", "Condition"],  # Needed for click detection
        category_orders={"domain": sorted_domains},
        color_discrete_map=color_map,
    )
    violin_fig.update_traces(jitter=0.75, selector=dict(mode="markers"))

    violin_fig.update_layout(
        title={"text": "Violin plot of DLS score values (sorted by effect size)", "x": 0.15},
        xaxis_title="Domain",
        yaxis_title="Z-score value",
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor="lightgray"),
        height=700,
        showlegend=True,
    )
    return violin_fig


def create_spider_plot(data, selected_sample, color_map=None):
    if color_map is None:
        color_map = COLOR_MAP
    
    # Ensure selected_sample is a string or None
    if selected_sample is not None and not isinstance(selected_sample, str):
        selected_sample = str(selected_sample)
        
    if not selected_sample:
        # Default spider plot when no sample is selected
        default_spider_fig = go.Figure()
        default_spider_fig.update_layout(
            title={
                "text": "Select a sample to display the corresponding radar plot",
                "y": 0.98,
                "x": 0.40,
                "xanchor": "center",
                "yanchor": "top",
            },
            font=dict(family="Arial, sans-serif", size=14, color="black"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=600,
            margin=dict(t=50, b=50, l=50, r=50),
        )
        return default_spider_fig

    # Handle average plots for different conditions
    if selected_sample.endswith(" average"):
        condition_name = selected_sample.replace(" average", "")
        available_conditions = data['Condition'].unique()
        
        # Check if the condition exists in the data
        if condition_name in available_conditions:
            return create_average_spider_plot(data, condition_name, color_map)
        
        # Fallback for original Control/CSDS naming
        if condition_name == "Control" and "Control" in available_conditions:
            return create_average_spider_plot(data, "Control", color_map)
        elif condition_name == "CSDS" and "CSDS" in available_conditions:
            return create_average_spider_plot(data, "CSDS", color_map)

    # Process data for individual sample radar plot
    data_radar = data.copy()
    data_radar["domain"] = pd.Categorical(
        data_radar["domain"],
        categories=["Socio-temporal functions", "Psychomotor changes", "Fatigue", "DLS score", "Lack of concentration", "Biological markers", "Apetite / weight", "Anxiety", "Anhedonia"],
    )
    data_radar = data_radar.sort_values("domain", ascending=True)
    data_radar["Z-score value"] = data_radar.groupby("domain", group_keys=False)[
        "Z-score value"
    ].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    filtered_df = data_radar[data_radar["Sample name"] == int(selected_sample)]
    filtered_df = filtered_df.loc[filtered_df.domain != "Total score"]

    if len(filtered_df) == 0:
        # Sample not found, return default plot instead of error
        default_spider_fig = go.Figure()
        default_spider_fig.update_layout(
            title={
                "text": "Sample not available in current filter",
                "y": 0.98,
                "x": 0.40,
                "xanchor": "center",
                "yanchor": "top",
            },
            font=dict(family="Arial, sans-serif", size=14, color="black"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=600,
            margin=dict(t=50, b=50, l=50, r=50),
        )
        return default_spider_fig

    condition = filtered_df["Condition"].iloc[0]
    categories = filtered_df["domain"].unique()
    mean_values = filtered_df.groupby("domain", group_keys=False)[
        "Z-score value"
    ].mean()

    spider_fig = px.line_polar(
        mean_values,
        r=mean_values.values,
        theta=categories,
        line_close=True,
    )

    spider_fig.update_layout(
        title={
            "text": f"Radar plot for sample {selected_sample}",
            "y": 0.98,
            "x": 0.25,
            "xanchor": "center",
            "yanchor": "top",
        },
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        plot_bgcolor="white",
        showlegend=True,
        height=600,
        margin=dict(t=100, b=100, l=100, r=100),
    )
    line_color = color_map.get(condition, color_discrete_sequence[0])
    spider_fig.update_traces(line=dict(color=line_color), fill="toself", line_width=5)
    spider_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.0, 1.0])),
        showlegend=False,
    )

    return spider_fig


def create_average_spider_plot(data, condition, color_map=None):
    """Create an average spider plot for a given condition (Control or CSDS)"""
    if color_map is None:
        color_map = COLOR_MAP
        
    # Process data for radar plot
    data_radar = data.copy()
    data_radar["domain"] = pd.Categorical(
        data_radar["domain"],
        categories=["Socio-temporal functions", "Psychomotor changes", "Fatigue", "DLS score", "Lack of concentration", "Biological markers", "Apetite / weight", "Anxiety", "Anhedonia"],
    )
    data_radar = data_radar.sort_values("domain", ascending=True)
    data_radar["Z-score value"] = data_radar.groupby("domain", group_keys=False)[
        "Z-score value"
    ].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    # Filter by condition and exclude Total score
    filtered_df = data_radar[data_radar["Condition"] == condition]
    filtered_df = filtered_df.loc[filtered_df.domain != "Total score"]
    
    # Calculate mean values for each domain
    mean_values = filtered_df.groupby("domain")["Z-score value"].mean()
    categories = mean_values.index.tolist()

    spider_fig = px.line_polar(
        mean_values,
        r=mean_values.values,
        theta=categories,
        line_close=True,
    )

    spider_fig.update_layout(
        title={
            "text": f"Average radar plot for {condition} condition",
            "y": 0.98,
            "x": 0.25,
            "xanchor": "center",
            "yanchor": "top",
        },
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        plot_bgcolor="white",
        showlegend=True,
        height=600,
        margin=dict(t=100, b=100, l=100, r=100),
    )
    
    line_color = color_map.get(condition, color_discrete_sequence[0])
    
    spider_fig.update_traces(line=dict(color=line_color), fill="toself", line_width=5)
    spider_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.0, 1.0])),
        showlegend=False,
    )

    return spider_fig


def intro_page():
    st.title("DZ Map - DLS Score Analysis Dashboard")

    st.markdown(
        """
    ## Introduction
    
    This dashboard provides a set of interactive visualizations for DLS (Depressive-Like Syndrome) scores across various domains, and allows researchers to submit their own data for analysis.
    
    ### Key Features:
    - **Data Table**: Browse the raw data directly in tabular form, and check that everything looks as expected.
    - **Data Exploration**: Explore the dataset with summary statistics and simple visualizations per domain.
    - **Score Radars**: Displays an interactive visualization with the distribution of scores per domain. Selecting a sample will show its profile across all domains as a radar plot.
    - **Clustered Heatmap**: Visualize patterns across samples and domains with optional hierarchical clustering.
    - **Vector Export**: All figures can be exported in high-quality vector formats (SVG, PDF) and raster formats (PNG).
    
    ### How to Use:
    1. The app loads the dataset presented in the accompanying paper by default.
    2. Alternatively, upload your data using the file uploader in the sidebar, as a CSV with domain scores as columns and animal IDs as rows.
    3. Navigate between tabs using the sidebar menu, alternating between several interactive visualizations.
    4. Under the "Score Radars" tab, select samples to view their detailed profiles by:
       - Using the dropdown menu
       - Clicking directly on points in the violin plot
    This will update the radar plot on the right.
    5. Use the download buttons below each figure to export in your preferred format.

    All generated figures can be downloaded for further analysis or publication.
    
    ### About the Data:
    The data represents Z-score values for various domains across different samples, categorized by condition (Control or DLS). Refer to the manuscript for more details on the methodology and analysis.
    """
    )

    st.markdown("---")
    st.markdown(
        """

    This tool accompanies the paper **Reinventing the wheel: Domain-focused biobehavioral assessment confirms a depression-like syndrome in C57BL/6 mice caused by chronic social defeat**.\n

    """
    )


def data_table_page(data):
    st.title("Data Overview")

    # Reconstruct the original format of the dataframe
    original_df = get_original_dataframe(data)

    st.markdown(
        """
    ### Raw Data Table
    
    This table shows the original dataset with samples as rows and domains as columns.
    You can:
    - Sort by clicking on column headers
    - Filter data using the search box
    - Download the data as CSV
    """
    )

    # Add search functionality
    search_term = st.text_input("Search samples", "")

    if original_df is not None:
        filtered_df = original_df
        if search_term:
            mask = original_df["Sample name"].str.contains(search_term, case=False)
            filtered_df = original_df[mask]

        # Add download button
        csv = filtered_df.to_csv()
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="dls_scores.csv",
            mime="text/csv",
        )

        # Display the dataframe
        st.dataframe(filtered_df, height=600)

        # Show summary statistics
        st.markdown("### Summary Statistics")
        st.markdown(f"**Total samples:** {original_df['Sample name'].nunique()}")
        st.markdown(f"**Conditions:** {', '.join(original_df['Condition'].unique())}")

        # Sample counts by condition
        condition_counts = (
            original_df.groupby("Condition").size().reset_index(name="count")
        )
        st.markdown("**Sample counts by condition:**")
        st.table(condition_counts)


def apply_custom_filters(data, filters_list):
    """Apply multiple custom filters to create binary classification based on domain thresholds"""
    if not filters_list:
        return data
    
    # Create a copy of the data
    filtered_data = copy.deepcopy(data)
    
    # For each sample, determine its new condition based on all filters
    sample_conditions = {}
    
    for sample in filtered_data['Sample name'].unique():
        sample_classifications = []
        
        for filter_info in filters_list:
            filter_domain = filter_info['domain']
            filter_threshold = filter_info['threshold']
            
            # Get the score for this sample in this domain
            domain_data = data[(data['domain'] == filter_domain) & (data['Sample name'] == sample)]
            if not domain_data.empty:
                sample_score = domain_data['Z-score value'].iloc[0]
                if sample_score >= filter_threshold:
                    sample_classifications.append('H')
                else:
                    sample_classifications.append('L')
            else:
                sample_classifications.append('?')  # Missing data
        
        # Create condition label based on all classifications
        if len(filters_list) == 1:
            # Single filter: use High/Low
            sample_conditions[sample] = 'High' if sample_classifications[0] == 'H' else 'Low'
        else:
            # Multiple filters: only show samples that fulfill all criteria or none
            if all(c == 'H' for c in sample_classifications):
                sample_conditions[sample] = 'High (all criteria)'
            elif all(c == 'L' for c in sample_classifications):
                sample_conditions[sample] = 'Low (all criteria)'
            elif any(c == 'H' for c in sample_classifications):
                sample_conditions[sample] = 'Mixed'
            else:
                # Skip mixed samples by not adding them to sample_conditions
                # This effectively filters them out of the visualization
                continue
    
    # Update the condition column based on the new classification
    filtered_data['Condition'] = filtered_data['Sample name'].map(sample_conditions)
    
    # Remove samples that don't have a condition assignment (mixed samples in multi-filter case)
    filtered_data = filtered_data.dropna(subset=['Condition'])
    
    return filtered_data


def apply_custom_filter(data, filter_domain, filter_threshold):
    """Apply single custom filter - wrapper for backwards compatibility"""
    if filter_domain is None:
        return data
    
    filters_list = [{'domain': filter_domain, 'threshold': filter_threshold}]
    return apply_custom_filters(data, filters_list)


def visualization_page(data):
    st.title("DLS Score Visualization")
    
    # Initialize session state variables if they don't exist
    if 'last_selection_method' not in st.session_state:
        st.session_state.last_selection_method = None
    if 'selected_sample' not in st.session_state:
        st.session_state.selected_sample = None
    if 'clicked_sample' not in st.session_state:
        st.session_state.clicked_sample = None
    if 'custom_filters_active' not in st.session_state:
        st.session_state.custom_filters_active = False
    if 'filters_list' not in st.session_state:
        st.session_state.filters_list = []
    
    # Custom Filter Section
    st.subheader("Custom Binary Classification")
    
    col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 1])
    
    with col_filter1:
        if st.button("Add Custom Filter", key="add_filter_btn"):
            st.session_state.custom_filters_active = True
    
    with col_filter2:
        if st.button("Reset All Filters", key="reset_filter_btn"):
            st.session_state.custom_filters_active = False
            st.session_state.filters_list = []
            st.session_state.selected_sample = None
            st.session_state.clicked_sample = None
    
    # Show filter controls if custom filter is active
    if st.session_state.custom_filters_active:
        st.markdown("**Configure Custom Filters:**")
        
        # Display existing filters
        if st.session_state.filters_list:
            st.markdown("**Active Filters:**")
            for i, filter_info in enumerate(st.session_state.filters_list):
                col_info, col_remove = st.columns([4, 1])
                with col_info:
                    st.write(f"{i+1}. {filter_info['domain']} ≥ {filter_info['threshold']:.1f}")
                with col_remove:
                    if st.button("Remove", key=f"remove_filter_{i}"):
                        st.session_state.filters_list.pop(i)
                        st.rerun()
        
        # Add new filter section
        st.markdown("**Add New Filter:**")
        col_domain, col_threshold, col_add = st.columns([3, 3, 1])
        
        with col_domain:
            # Get available domains (excluding 'Total score' and 'DLS score')
            available_domains = [d for d in data['domain'].unique() 
                               if d not in ['Total score']]
            
            new_filter_domain = st.selectbox(
                "Select Domain",
                options=[None] + available_domains,
                key="new_filter_domain_select",
                format_func=lambda x: "Choose domain..." if x is None else x
            )
        
        with col_threshold:
            if new_filter_domain:
                # Get min and max values for the selected domain
                domain_data = data[data['domain'] == new_filter_domain]
                min_val = float(domain_data['Z-score value'].min())
                max_val = float(domain_data['Z-score value'].max())
                mean_val = float(domain_data['Z-score value'].mean())
                
                new_filter_threshold = st.slider(
                    f"Threshold for {new_filter_domain}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=0.1,
                    key="new_filter_threshold_slider"
                )
                
                # Show classification info for this domain
                high_count = len(domain_data[domain_data['Z-score value'] >= new_filter_threshold])
                low_count = len(domain_data[domain_data['Z-score value'] < new_filter_threshold])
                st.info(f"**High** (≥{new_filter_threshold:.1f}): {high_count} | **Low** (<{new_filter_threshold:.1f}): {low_count}")
        
        with col_add:
            if new_filter_domain is not None:
                st.write("")  # Add spacing
                st.write("")  # Add spacing
                if st.button("Add Filter", key="add_new_filter_btn"):
                    # Check if this domain is already in the filters
                    existing_domains = [f['domain'] for f in st.session_state.filters_list]
                    if new_filter_domain not in existing_domains:
                        st.session_state.filters_list.append({
                            'domain': new_filter_domain,
                            'threshold': new_filter_threshold
                        })
                        st.rerun()
                    else:
                        st.warning(f"Filter for {new_filter_domain} already exists!")
        
        # Show combined filter information
        if st.session_state.filters_list:
            st.markdown("**Combined Classification Preview:**")
            # Apply filters to show classification counts
            preview_data = apply_custom_filters(data, st.session_state.filters_list)
            condition_counts = preview_data.groupby("Sample name")["Condition"].first().value_counts()
            
            # Display in columns
            if len(condition_counts) > 0:
                cols = st.columns(len(condition_counts))
                for i, (condition, count) in enumerate(condition_counts.items()):
                    with cols[i]:
                        st.metric(condition, count)
    
    # Apply custom filters if active
    if st.session_state.custom_filters_active and st.session_state.filters_list:
        display_data = apply_custom_filters(data, st.session_state.filters_list)
        
        # Create dynamic color map based on unique conditions
        unique_conditions = display_data['Condition'].unique()
        current_color_map = {}
        
        # Assign colors from the palette
        for i, condition in enumerate(unique_conditions):
            current_color_map[condition] = color_discrete_sequence[i % len(color_discrete_sequence)]
        
        # Create filter info string
        filter_descriptions = [f"{f['domain']} ≥ {f['threshold']:.1f}" for f in st.session_state.filters_list]
        filter_info = f" (Filtered by: {'; '.join(filter_descriptions)})"
    else:
        display_data = data
        current_color_map = COLOR_MAP
        filter_info = ""
    
    # Get sample options and add average options
    sample_options = sorted(display_data["Sample name"].unique())
    
    # Create average options based on unique conditions
    unique_conditions = display_data['Condition'].unique()
    average_options = [f"{condition} average" for condition in unique_conditions]
    dropdown_options = average_options + sample_options
    
    # Define callback for dropdown selection
    def on_dropdown_change():
        st.session_state.clicked_sample = None
        st.session_state.last_selection_method = "dropdown"
    
    # Dropdown for manual selection
    selected_sample = st.selectbox(
        "Click a sample on the plot on the left, or select manually from the drop-down menu",
        options=[None] + dropdown_options,
        format_func=lambda x: "Select a sample" if x is None else x,
        key="dropdown_selection",
        on_change=on_dropdown_change
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create violin plot with filtered data
        violin_fig = create_violin_plot(display_data, color_map=current_color_map)
        violin_fig.update_layout(title={"text": f"Violin plot of score values (sorted by effect size){filter_info}", "x": 0.15})
        
        # Display violin plot with sample highlighted
        event_dict = st.plotly_chart(violin_fig, theme="streamlit", use_container_width=True, on_select="rerun", selection_mode="points")
        
        # Add download buttons for violin plot
        st.markdown("**Export Violin Plot:**")
        create_download_buttons_row_with_csv(
            violin_fig, 
            "violin_plot", 
            "violin", 
            display_data, 
            "violin_plot_data.csv"
        )
        
        # Extract sample from click event
        try:
            clicked_sample = event_dict['selection']['points'][0]['customdata'][0]
            # Check if the clicked sample exists in the filtered data
            if clicked_sample is not None and clicked_sample in display_data['Sample name'].values:
                st.session_state.clicked_sample = clicked_sample
                st.session_state.last_selection_method = "plot_click"
            else:
                # Sample was filtered out, clear the selection
                st.session_state.clicked_sample = None
                if clicked_sample is not None:
                    st.warning(f"Sample '{clicked_sample}' was excluded due to filtering criteria.")
        except (KeyError, IndexError):
            # No click event occurred
            pass
    
    # Determine which sample to display based on last selection method
    if st.session_state.last_selection_method == "plot_click":
        final_sample = st.session_state.clicked_sample
    else:
        final_sample = selected_sample
    
    # Validate that the final sample exists in the filtered data (if it's not an average)
    if (final_sample is not None and 
        not str(final_sample).endswith(" average") and 
        final_sample not in display_data['Sample name'].values):
        final_sample = None

    with col2:
        spider_fig = create_spider_plot(display_data, selected_sample=final_sample, color_map=current_color_map)
        st.plotly_chart(spider_fig, use_container_width=True)
        
        # Add download buttons for spider plot (only if a sample is selected)
        if final_sample:
            st.markdown("**Export Radar Plot:**")
            # Prepare data for the specific sample or average
            if str(final_sample).endswith(" average"):
                condition_name = str(final_sample).replace(" average", "")
                radar_data = display_data[display_data['Condition'] == condition_name].copy()
                csv_filename = f"radar_plot_{condition_name}_average_data.csv"
            else:
                radar_data = display_data[display_data['Sample name'] == final_sample].copy()
                csv_filename = f"radar_plot_sample_{final_sample}_data.csv"
            
            create_download_buttons_row_with_csv(
                spider_fig, 
                f"radar_plot_{final_sample}", 
                f"radar_{final_sample}",
                radar_data,
                csv_filename
            )

    
def data_analysis_page(data):
    st.title("Data Analysis")

    st.markdown("### Overview of the Dataset")

    # Show basic statistics
    st.subheader("Basic Statistics")
    st.write(f"Total number of samples: {data['Sample name'].nunique()}")
    st.write(f"Number of domains: {data['domain'].nunique()}")

    # Sample distribution by condition
    st.subheader("Sample Distribution by Condition")
    condition_counts = data.groupby("Sample name")["Condition"].first().value_counts()

    condition_fig = px.pie(
        values=condition_counts.values,
        names=condition_counts.index,
        title="Distribution of Samples by Condition",
    )
    # Manually set the colors to ensure Set2 palette is used
    condition_fig.update_traces(
        marker=dict(colors=[COLOR_MAP.get(name, color_discrete_sequence[i]) 
                        for i, name in enumerate(condition_counts.index)])
    )
    st.plotly_chart(condition_fig)
    
    # Add download buttons for pie chart
    st.markdown("**Export Pie Chart:**")
    # Prepare pie chart data
    pie_data = pd.DataFrame({
        'Condition': condition_counts.index,
        'Count': condition_counts.values
    })
    create_download_buttons_row_with_csv(
        condition_fig, 
        "sample_distribution_pie", 
        "pie",
        pie_data,
        "sample_distribution_data.csv"
    )

    # Show domain summary statistics
    st.subheader("Domain Statistics")
    domain_stats = (
        data.groupby(["domain", "Condition"])["Z-score value"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    st.dataframe(domain_stats)

    # Show a heatmap of domain correlations (excluding DLS score)
    st.subheader("Domain Correlation Heatmap")
    # Pivot data to get domains as columns
    pivot_data = data.pivot_table(
        index="Sample name", columns="domain", values="Z-score value", aggfunc="mean"
    )
    
    # Remove DLS score column from correlation analysis
    if "DLS score" in pivot_data.columns:
        pivot_data = pivot_data.drop("DLS score", axis=1)

    # Calculate correlation matrix
    corr_matrix = pivot_data.corr()

    # Create clustered correlation heatmap using seaborn
    clustered_corr = sns.clustermap(
        corr_matrix,
        annot=True,
        fmt='.2f',  # Format annotations to 2 decimal places
        cmap='viridis',
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        figsize=(12, 10),
    )
    
    # Save to buffer for display
    corr_buffer = io.BytesIO()
    clustered_corr.savefig(corr_buffer, format='png', bbox_inches='tight', dpi=300)
    corr_buffer.seek(0)
    
    # Display the clustered correlation heatmap
    st.image(corr_buffer, use_container_width=False)
    
    # Create download buttons for clustered correlation heatmap
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        corr_buffer.seek(0)
        st.download_button(
            label="🖼️ Download PNG",
            data=corr_buffer,
            file_name="clustered_correlation_heatmap.png",
            mime="image/png",
            key="corr_heatmap_png"
        )
    
    with col2:
        corr_svg_buffer = io.BytesIO()
        clustered_corr.savefig(corr_svg_buffer, format='svg', bbox_inches='tight')
        corr_svg_buffer.seek(0)
        st.download_button(
            label="📊 Download SVG",
            data=corr_svg_buffer,
            file_name="clustered_correlation_heatmap.svg",
            mime="image/svg+xml",
            key="corr_heatmap_svg"
        )
    
    with col3:
        corr_pdf_buffer = io.BytesIO()
        clustered_corr.savefig(corr_pdf_buffer, format='pdf', bbox_inches='tight')
        corr_pdf_buffer.seek(0)
        st.download_button(
            label="📄 Download PDF",
            data=corr_pdf_buffer,
            file_name="clustered_correlation_heatmap.pdf",
            mime="application/pdf",
            key="corr_heatmap_pdf"
        )
    
    with col4:
        # Download correlation matrix data
        corr_csv = corr_matrix.to_csv()
        st.download_button(
            label="📋 Download CSV",
            data=corr_csv,
            file_name="correlation_matrix_data.csv",
            mime="text/csv",
            key="corr_heatmap_csv"
        )
    
    plt.close('all')  # Clean up matplotlib figures
    
    # Download buttons are now integrated above


def clustered_heatmap_page(data):
    st.title("Clustered Heatmap")

    st.markdown(
        """
    ### Hierarchical Clustered Heatmap
    
    This heatmap shows the Z-scores across all domains and samples, with:
    - Hierarchical clustering of both domains (rows) and samples (columns)
    - Color-coding of samples by condition (Control vs. DLS)
    - Z-score normalization for better comparison
    """
    )

    # Create a pivot table with samples as columns and domains as rows
    original_df = get_original_dataframe(data)

    if original_df is not None:
        # Prepare data for clustering
        # Reshape to samples as columns, domains as rows
        domain_scores = original_df.copy()

        # Display options
        st.sidebar.subheader("Heatmap Options")
        cluster_rows = st.sidebar.checkbox("Cluster Rows (Domains)", value=True)
        cluster_cols = st.sidebar.checkbox("Cluster Columns (Samples)", value=True)
        z_score_normalize = st.sidebar.checkbox("Z-score Normalization", value=False)
        cmap_option = st.sidebar.selectbox(
            "Color Map",
            options=["viridis", "coolwarm", "plasma", "inferno", "magma", "cividis"],
            index=0,  # viridis is now the default
        )

        # Create the heatmap using matplotlib
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set up sample colors for the color bar using Set2 colors
        sample_colors = domain_scores["Condition"].map(COLOR_MAP)

        # Prepare data matrix for heatmap
        data_matrix = domain_scores.set_index(["Sample name", "Condition"])
        domains = [col for col in data_matrix.columns if col != "DLS score"]
        data_matrix = data_matrix[domains].T

        # Apply Z-score normalization if selected
        if z_score_normalize:
            data_matrix = (data_matrix - data_matrix.mean()) / data_matrix.std()

        # Generate heatmap using seaborn's clustermap
        buffer = io.BytesIO()

        try:
            # Create the clustermap
            sns.set_context("paper")
            heatmap = sns.clustermap(
                data_matrix,
                figsize=(24, 10),
                cmap=cmap_option,
                z_score=0 if z_score_normalize else None,
                row_cluster=cluster_rows,
                col_cluster=cluster_cols,
                col_colors=pd.Series(sample_colors.values, index=data_matrix.columns),
                cbar_pos=(-0.1, 0.2, 0.03, 0.4),
            )

            # Remove X ticks and labels
            ax_heatmap = heatmap.ax_heatmap
            ax_heatmap.set_xticks([])
            ax_heatmap.set_xlabel("Animal")

            # Save figure to buffer
            plt.savefig(buffer, format="png", bbox_inches="tight", dpi=300)
            buffer.seek(0)

            # Display the figure using st.image
            st.image(buffer, use_container_width=True, output_format="PNG")

            # Add download buttons for the heatmap
            st.markdown("**Export Clustered Heatmap:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # PNG download
                buffer.seek(0)
                st.download_button(
                    label="🖼️ Download PNG",
                    data=buffer,
                    file_name="clustered_heatmap.png",
                    mime="image/png",
                    key="heatmap_png"
                )
            
            with col2:
                # SVG download
                svg_buffer = io.BytesIO()
                plt.savefig(svg_buffer, format="svg", bbox_inches="tight")
                svg_buffer.seek(0)
                st.download_button(
                    label="📊 Download SVG",
                    data=svg_buffer,
                    file_name="clustered_heatmap.svg",
                    mime="image/svg+xml",
                    key="heatmap_svg"
                )
            
            with col3:
                # PDF download
                pdf_buffer = io.BytesIO()
                plt.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
                pdf_buffer.seek(0)
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_buffer,
                    file_name="clustered_heatmap.pdf",
                    mime="application/pdf",
                    key="heatmap_pdf"
                )
            
            with col4:
                # Download heatmap data
                heatmap_csv = data_matrix.to_csv()
                st.download_button(
                    label="📋 Download CSV",
                    data=heatmap_csv,
                    file_name="clustered_heatmap_data.csv",
                    mime="text/csv",
                    key="heatmap_csv"
                )

        except Exception as e:
            st.error(f"Error generating heatmap: {e}")
            st.text(
                "Please ensure your data is suitable for clustering (no missing values, sufficient samples)."
            )
    else:
        st.warning("No data available to generate heatmap.")


def main():
    # Add logo or title to sidebar
    st.sidebar.title("DLS Score Dashboard")
    st.sidebar.markdown("---")

    # File upload in sidebar
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    # Load data (with default option)
    data = load_data(uploaded_file)

    # If no uploaded file and default file exists, load it
    if uploaded_file is None and data is None:
        file_exists = os.path.exists(DEFAULT_FILE)
        if file_exists:
            st.sidebar.info(f"Using default dataset: {DEFAULT_FILE}")
            data = load_data(DEFAULT_FILE)
        else:
            st.sidebar.warning(
                f"Default file {DEFAULT_FILE} not found. Please upload a file."
            )

    if data is not None:
        # Navigation menu
        page = st.sidebar.radio(
            "Navigation",
            options=[
                "Introduction",
                "Data Table",
                "Data Exploration",
                "Score Radars",
                "Clustered Heatmap",
            ],
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("Lucas Miranda, MLSB 2025")

        # Display the selected page
        if page == "Introduction":
            intro_page()
        elif page == "Data Table":
            data_table_page(data)
        elif page == "Score Radars":
            visualization_page(data)
        elif page == "Data Exploration":
            data_analysis_page(data)
        elif page == "Clustered Heatmap":
            clustered_heatmap_page(data)
    else:
        # Show instructions if no data is loaded
        st.title("Welcome to the DLS Score Dashboard")
        st.markdown(
            """
        ### Getting Started
        
        To use this dashboard, you need to provide data in CSV format. You can:
        
        1. Place a file named `DLS_scores.csv` in the same directory as this app
        2. Upload a CSV file using the uploader in the sidebar
        3. Run the app with a file path as argument: `streamlit run app.py -- path_to_file.csv`
        
        The CSV should contain:
        - A column named "Sample name" with unique identifiers
        - A column named "Condition" with categories (e.g., "Control", "DLS")
        - Additional columns for each domain with numeric Z-score values
        """
        )


if __name__ == "__main__":
    main()