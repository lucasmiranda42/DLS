import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
from sys import argv
import seaborn as sns

palette = sns.color_palette("Set2", as_cmap=True)
color_discrete_sequence = sns.color_palette("Set2").as_hex()


def run_dashboard(data):
    app = dash.Dash(__name__)

    # Create dropdown options from unique sample names
    sample_options = [
        {"label": sample, "value": sample}
        for sample in sorted(data["Sample name"].unique())
    ]

    app.layout = html.Div(
        [
            html.Div(
                [
                    # Add dropdown for sample selection
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="sample-dropdown",
                                options=sample_options,
                                value=None,
                                placeholder="Choose a sample or click on the violin plot",
                                style={
                                    "width": "400px",
                                    "font-family": "Arial, sans-serif",
                                    "font-size": "14px",
                                },
                            )
                        ],
                        style={
                            "margin-bottom": "10px",
                            "margin-left": "50px",
                            "display": "flex",
                            "align-items": "center",
                            "justify-content": "left",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="violin-graph", style={"height": "90vh"}
                                    ),
                                ],
                                style={
                                    "width": "50%",
                                    "display": "inline-block",
                                    "height": "90vh",
                                },
                            ),
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="spider-graph", style={"height": "90vh"}
                                    ),
                                ],
                                style={
                                    "width": "50%",
                                    "display": "inline-block",
                                    "height": "90vh",
                                },
                            ),
                        ],
                        style={
                            "width": "100%",
                            "display": "inline-block",
                            "height": "90vh",
                        },
                    ),
                ]
            ),
            html.Div(
                "Lucas Miranda, MLSB 2025",
                id="watermark",
                style={"color": "gray", "font-size": "15px", "margin-top": "100px"},
            ),
        ]
    )

    @app.callback(
        [Output("violin-graph", "figure"), Output("spider-graph", "figure")],
        [Input("violin-graph", "clickData"), Input("sample-dropdown", "value")],
    )
    def update_graphs(clickData, dropdown_sample):
        # Determine the selected sample
        if dropdown_sample:
            selected_sample = dropdown_sample
        elif clickData:
            selected_sample = clickData["points"][0]["customdata"][0]
        else:
            # Default state when no selection is made
            violin_fig = px.strip(
                data,
                y="Z-score value",
                x="domain",
                color="Condition",
                hover_data={"Sample name": True},
                custom_data=["Sample name", "Condition"],
                category_orders={
                    "domain": list(
                        data.sort_values(by=["Z-score value"]).domain.unique()
                    )
                },
                color_discrete_map={
                    "Control": color_discrete_sequence[0],
                    "DLS": color_discrete_sequence[1],
                },
            )
            violin_fig.update_layout(
                title={
                    "text": "Violin plot of DLS score values",
                    "y": 0.98,
                    "x": 0.15,
                    "xanchor": "center",
                    "yanchor": "top",
                },
                xaxis_title="Domain",
                yaxis_title="Z-score value",
                font=dict(family="Arial, sans-serif", size=14, color="black"),
                plot_bgcolor="white",
                xaxis=dict(showgrid=False, tickangle=-45),
                yaxis=dict(showgrid=True, gridcolor="lightgray"),
                showlegend=True,
            )
            violin_fig.update_traces(jitter=0.75)

            default_spider_fig = go.Figure()
            default_spider_fig.update_layout(
                title={
                    "text": "Select a sample to display the corresponding radar plot",
                    "y": 0.98,
                    "x": 0.25,
                    "xanchor": "center",
                    "yanchor": "top",
                },
                font=dict(family="Arial, sans-serif", size=14, color="black"),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            return violin_fig, default_spider_fig

        # Violin plot
        violin_fig = px.strip(
            data,
            y="Z-score value",
            x="domain",
            color="Condition",
            hover_data={"Sample name": True},
            custom_data=["Sample name", "Condition"],
            category_orders={
                "domain": list(data.sort_values(by=["Z-score value"]).domain.unique())
            },
            color_discrete_map={
                "Control": color_discrete_sequence[0],
                "DLS": color_discrete_sequence[1],
            },
        )
        violin_fig.update_layout(
            title={
                "text": "Violin plot of DLS score values",
                "y": 0.98,
                "x": 0.15,
                "xanchor": "center",
                "yanchor": "top",
            },
            xaxis_title="Domain",
            yaxis_title="Z-score value",
            font=dict(family="Arial, sans-serif", size=14, color="black"),
            plot_bgcolor="white",
            xaxis=dict(showgrid=False, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor="lightgray"),
            showlegend=True,
        )
        violin_fig.update_traces(jitter=0.75)

        # Highlight selected sample
        selected_points = data[data["Sample name"] == selected_sample]
        violin_fig.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.2)))
        violin_fig.add_trace(
            go.Scatter(
                x=selected_points["domain"],
                y=selected_points["Z-score value"],
                mode="markers",
                marker=dict(
                    size=10,
                    color="black",
                    line=dict(color="black", width=3),
                    symbol="circle-open",
                ),
                name="Selected sample",
                hoverinfo="skip",
            )
        )

        # Spider plot
        data_radar = data.copy()
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
                "x": 0.15,
                "xanchor": "center",
                "yanchor": "top",
            },
            font=dict(family="Arial, sans-serif", size=14, color="black"),
            plot_bgcolor="white",
            showlegend=True,
        )
        line_color = {
            "Control": color_discrete_sequence[0],
            "DLS": color_discrete_sequence[1],
        }[condition]
        spider_fig.update_traces(
            line=dict(color=line_color), fill="toself", line_width=5
        )
        spider_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.0, 1.0])),
            showlegend=False,
        )

        return violin_fig, spider_fig

    app.run_server(debug=True)


if __name__ == "__main__":
    try:
        scores_per_domain = pd.read_csv(argv[1], index_col=0)
    except IndexError:
        print("Please provide path to the scores per domain file.")
        exit()

    scores_per_domain = scores_per_domain.melt(
        id_vars=["Sample name", "Condition"],
        var_name="domain",
        value_name="Z-score value",
    ).sort_values(by=["Condition", "domain", "Z-score value"])

    run_dashboard(scores_per_domain)
