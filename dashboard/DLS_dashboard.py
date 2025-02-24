# Import necessary libraries
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
from sys import argv


def run_dashboard(data):

    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            # Main Row
            html.Div(
                [
                    # Column for Graphs
                    html.Div(
                        [
                            # Panel for Violin Graph
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="violin-graph", style={"height": "90vh"}
                                    ),
                                    # Set the height of the graph
                                ],
                                style={
                                    "width": "50%",
                                    "display": "inline-block",
                                    "height": "90vh",
                                },
                                # Set the height of the container
                            ),
                            # Panel for Spider Graph
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="spider-graph", style={"height": "90vh"}
                                    ),
                                    # Set the height of the graph
                                ],
                                style={
                                    "width": "50%",
                                    "display": "inline-block",
                                    "height": "90vh",
                                },
                                # Set the height of the container
                            ),
                        ],
                        style={"width": "100%", "display": "flex", "height": "90vh"},
                        # Ensure the parent container also has a large enough height
                    ),
                ],
            ),
            # Watermark
            html.Div(
                "Lucas Miranda, MLSB 2023",
                id="watermark",
                style={"color": "gray", "font-size": "20px", "margin-top": "70px"},
            ),
            # Adjust the watermark position
        ],
    )

    @app.callback(
        Output("spider-graph", "figure"),
        [Input("violin-graph", "clickData")],
        [dash.dependencies.State("violin-graph", "figure")],
    )
    def update_radar_chart(clickData, violin_figure):

        if clickData:

            # Get the selected sample
            selected_sample = clickData["points"][0]["customdata"][0]

            # Normalize data so that each domain is between 0 and 1
            data_radar = data.copy()
            data_radar["Z-score value"] = data_radar.groupby("domain")["Z-score value"].apply(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )

            filtered_df = data_radar[data_radar["Sample name"] == selected_sample]
            filtered_df = filtered_df.loc[filtered_df.domain != "Total score"]

            # Check the condition of the selected sample
            condition = filtered_df["Condition"].iloc[0]

            # Spider plot
            categories = filtered_df["domain"].unique()
            mean_values = filtered_df.groupby("domain")["Z-score value"].mean()
            spider_fig = px.line_polar(
                mean_values,
                r=mean_values.values,
                theta=categories,
                line_close=True,
                title=f"Sample {selected_sample}",
            )

            # Update line color based on the condition
            line_color = "red" if condition == "Stressed" else "blue"
            spider_fig.update_traces(line=dict(color=line_color), fill="toself")
            spider_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0.0, 1.0], # Tune accordingly
                    )),
                showlegend=False
            )

            return spider_fig
        else:
            # Return an empty radar chart
            return go.Figure()

    # Initial Plot
    @app.callback(
        Output("violin-graph", "figure"),
        [Input("violin-graph", "clickData")]
    )
    def update_violin_chart(clickData):
        # Generate initial strip plot
        violin_fig = px.strip(
            data,
            y="Z-score value",
            x="domain",
            color="Condition",
            hover_data=["Sample name"],
            custom_data=["Sample name"],
            category_orders={
                "domain": list(data.sort_values(by=["Z-score value"]).domain.unique())
            },
        )

        violin_fig.update_traces(marker=dict(size=8))  # Adjust the size of original points if needed

        violin_fig.update_layout(
            xaxis_title="DLS domain",
            yaxis_title="Z-score",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        if clickData:
            # Get the selected sample name
            selected_sample = clickData["points"][0]["customdata"][0]

            # Find all points with the selected sample name
            selected_points = data[data["Sample name"] == selected_sample]

            # Overlay these points on the strip plot
            violin_fig.add_trace(
                go.Scatter(
                    x=selected_points['domain'],
                    y=selected_points['Z-score value'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='black',
                        line=dict(
                            color='black',  # outline color
                            width=5  # outline width
                        ),
                        symbol='circle-open'  # open circle symbol
                    ),
                    name='Selected sample',
                    hoverinfo='skip'  # to avoid hover on the highlighted points if desired
                )
            )

        return violin_fig

    app.run_server(debug=True)


if __name__ == "__main__":

    # Read data
    try:
        scores_per_domain = pd.read_csv(argv[1], index_col=0)
    except IndexError:
        print("Please provide path to the scores per domain file.")
        exit()

    # Melt data
    scores_per_domain = scores_per_domain.melt(
        id_vars=["Sample name", "Condition"],
        var_name="domain",
        value_name="Z-score value",
    ).sort_values(by=["Condition", "domain", "Z-score value"])

    # Run dashboard
    run_dashboard(scores_per_domain)
