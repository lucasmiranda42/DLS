import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io
import os

from sys import argv


# Set page configuration with light theme
st.set_page_config(
    page_title="DZMap - DLS Score Analysis", page_icon="📊", layout="wide"
)

# Set up color palette
palette = sns.color_palette("Set2", as_cmap=True)
color_discrete_sequence = sns.color_palette("Set2").as_hex()

# Set default file path
DEFAULT_FILE = "../DLS_data/Results/DLS_domain_scores.csv"


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

    return scores_per_domain


def get_original_dataframe(melted_data):
    """Reconstruct the original dataframe format from the melted data"""
    if melted_data is not None:
        original_df = melted_data.pivot_table(
            index=["Sample name", "Condition"], columns="domain", values="Z-score value"
        ).reset_index()
        return original_df
    return None


def create_violin_plot(data):
    violin_fig = px.strip(
        data,
        y="Z-score value",
        x="domain",
        color="Condition",
        hover_data={"Sample name": True},
        custom_data=["Sample name", "Condition"],  # Needed for click detection
        category_orders={
            "domain": list(data.sort_values(by=["Z-score value"]).domain.unique())
        },
        color_discrete_map={
            "Control": color_discrete_sequence[0],
            "DLS": color_discrete_sequence[1],
        },
    )
    violin_fig.update_traces(jitter=0.75, selector=dict(mode="markers"))

    violin_fig.update_layout(
        title={"text": "Violin plot of DLS score values", "x": 0.15},
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


def create_spider_plot(data, selected_sample):
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
    filtered_df = data_radar[data_radar["Sample name"] == selected_sample]
    filtered_df = filtered_df.loc[filtered_df.domain != "Total score"]

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
    line_color = {
        "Control": color_discrete_sequence[0],
        "DLS": color_discrete_sequence[1],
    }[condition]
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
    
    This dashboard provides an interactive visualization of DLS (Depressive-Like Syndrome) scores across various domains.
    
    ### Key Features:
    - **Violin Plot**: Displays the distribution of Z-score values across domains, colored by condition (Control vs. DLS).
    - **Radar Plot**: Shows the normalized profile of a selected sample across all domains.
    - **Sample Selection**: Select samples either from the dropdown menu or by clicking directly on the violin plot.
    - **Data Table**: Browse the raw data in tabular form.
    - **Clustered Heatmap**: Visualize patterns across samples and domains with hierarchical clustering.
    
    ### How to Use:
    1. The app loads a default dataset (DLS_scores.csv) if available
    2. Alternatively, upload your data using the file uploader in the sidebar
    3. Navigate between tabs using the sidebar menu
    4. Select samples to view their detailed profiles by:
       - Using the dropdown menu
       - Clicking directly on points in the violin plot
    
    ### About the Data:
    The data represents Z-score values for various domains across different samples, categorized by condition (Control or DLS).
    """
    )

    st.markdown("---")
    st.markdown("### Additional Information")
    st.markdown(
        """
    You can add more information about your project, methodology, or any other relevant context here.
    
    This section can be customized to include:
    - Research objectives
    - Experimental design
    - Data collection methods
    - Analysis protocols
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


def visualization_page(data):
    st.title("DLS Score Visualization")
    
    # Initialize session state variables if they don't exist
    if 'last_selection_method' not in st.session_state:
        st.session_state.last_selection_method = None
    if 'selected_sample' not in st.session_state:
        st.session_state.selected_sample = None
    if 'clicked_sample' not in st.session_state:
        st.session_state.clicked_sample = None
    
    sample_options = sorted(data["Sample name"].unique())
    
    # Define callback for dropdown selection
    def on_dropdown_change():
        st.session_state.clicked_sample = None
        st.session_state.last_selection_method = "dropdown"
    
    # Dropdown for manual selection
    selected_sample = st.selectbox(
        "Click a sample on the plot on the left, or select manually from the drop-down menu",
        options=[None] + sample_options,
        format_func=lambda x: "Select a sample" if x is None else x,
        key="dropdown_selection",
        on_change=on_dropdown_change
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create violin plot
        base_violin_fig = create_violin_plot(data)
        # Display violin plot with sample highlighted
        event_dict = st.plotly_chart(base_violin_fig, theme="streamlit", use_container_width=True, on_select="rerun", selection_mode="points")
        # Extract sample from click event
        try:
            clicked_sample = event_dict['selection']['points'][0]['customdata'][0]
            # Update session state when plot is clicked
            if clicked_sample is not None:
                st.session_state.clicked_sample = clicked_sample
                st.session_state.last_selection_method = "plot_click"
        except (KeyError, IndexError):
            # No click event occurred
            pass
    
    # Determine which sample to display based on last selection method
    if st.session_state.last_selection_method == "plot_click":
        final_sample = st.session_state.clicked_sample
    else:
        final_sample = selected_sample

    with col2:
        spider_fig = create_spider_plot(data, selected_sample=final_sample)
        st.plotly_chart(spider_fig, use_container_width=True)

    
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
        color_discrete_sequence=color_discrete_sequence,
        title="Distribution of Samples by Condition",
    )
    st.plotly_chart(condition_fig)

    # Show domain summary statistics
    st.subheader("Domain Statistics")
    domain_stats = (
        data.groupby(["domain", "Condition"])["Z-score value"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    st.dataframe(domain_stats)

    # Show a heatmap of domain correlations
    st.subheader("Domain Correlation Heatmap")
    # Pivot data to get domains as columns
    pivot_data = data.pivot_table(
        index="Sample name", columns="domain", values="Z-score value", aggfunc="mean"
    )

    # Calculate correlation matrix
    corr_matrix = pivot_data.corr()

    # Create heatmap
    heatmap_fig = px.imshow(
        corr_matrix,
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Correlation Between Domains",
    )
    heatmap_fig.update_layout(height=600)
    st.plotly_chart(heatmap_fig)


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
        z_score_normalize = st.sidebar.checkbox("Z-score Normalization", value=True)
        cmap_option = st.sidebar.selectbox(
            "Color Map",
            options=["coolwarm", "viridis", "plasma", "inferno", "magma", "cividis"],
            index=0,
        )

        # Create the heatmap using matplotlib
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set up sample colors for the color bar
        sample_colors = domain_scores["Condition"].map(
            {"Control": "blue", "DLS": "red"}
        )

        # Prepare data matrix for heatmap
        data_matrix = domain_scores.set_index(["Sample name", "Condition"])
        domains = [col for col in data_matrix.columns if col != "Total score"]
        data_matrix = data_matrix[domains].T

        # Apply Z-score normalization if selected
        if z_score_normalize:
            data_matrix = (data_matrix - data_matrix.mean()) / data_matrix.std()

        # Generate heatmap using seaborn's clustermap
        buffer = io.BytesIO()

        try:
            # Create the clustermap
            heatmap = sns.clustermap(
                data_matrix,
                figsize=(12, 8),
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
            st.image(buffer, use_container_width=True)

            # Add download button for the heatmap
            btn = st.download_button(
                label="Download Heatmap",
                data=buffer,
                file_name="clustered_heatmap.png",
                mime="image/png",
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
