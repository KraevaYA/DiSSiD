# coding: utf-8

import numpy as np
import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
#plotly.offline.init_notebook_mode(connected=True)

from config import PLOTS_DIR


def plot_ts(ts, n, plot_path, title = "Input Time Series"):
    """
    Plot the time series
    """
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(np.arange(n)), y=list(ts)))
    
    fig.update_xaxes(showgrid=False,
                     title='Time',
                     title_font=dict(size=14, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=14, color='black'),
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)
        
    fig.update_yaxes(showgrid=False,
                     title='Values',
                     title_font=dict(size=14, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=14, color='black'),
                     zeroline=False,
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)
    
    fig.update_layout(title=title,
                      title_font=dict(size=16, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      showlegend=False,
                      margin=dict(l=10, r=10, t=40, b=10),
                      width=1000, height=300
                      )

    fig.write_image(plot_path, scale=3)


def plot_discords(ts, mp, discords, n, m, N, discords_num, plot_path):
    """
    Plot top-k discords in time series and Matrix Profile
    """
    
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("Time Series", "Matrix Profile with Top-k discords")
                        )
        
    # plot time series with discords
    fig.add_trace(go.Scatter(x=list(np.arange(n)),
                             y=list(ts),
                             line=dict(color='#636EFA', width=2),
                             name="Time Series"),
                  row=1, col=1)
    
    # plot mp and discords
    discords_mp = [mp[discord_ind] for discord_ind in discords]
    fig.add_trace(go.Scatter(x=list(np.arange(len(mp))),
                             y=list(mp),
                             line=dict(color='#636EFA', width=2),
                             name="Matrix Profile"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=list(discords),
                             y=discords_mp, mode='markers',
                             marker=dict(symbol='star', color='red', size=7),
                             name="Discords"),
                  row=2, col=1)
    
    fig.update_annotations(font=dict(size=16, color='black'))
    
    fig.update_xaxes(showgrid=False,
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=14, color='black'),
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)

    fig.update_yaxes(showgrid=False,
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=14, color='black'),
                     zeroline=False,
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)
                     
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      showlegend=False,
                      margin=dict(l=10, r=10, t=30, b=10),
                      width=1000, height=500,
                      )

    fig.write_image(plot_path, scale=3)


def plot_snippets(ts, snippets, n, m, snippets_num, plot_path):
    """
    Plot snippets
    """
    
    #len_profile = len(snippets['profiles'][0])
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#8C564B', '#316395', '#BCBD22', '#7F7F7F', '#2CA02C', '#AF0038']
    
    fig = make_subplots(rows=3, cols=1,
                        vertical_spacing=0.05)
    
    # plot time series with snippets
    fig.add_trace(go.Scatter(x=np.arange(n), y=list(ts), name="Time Series", line=dict(color=colors[0])), row=1, col=1)
    for i in range(snippets_num):
        snippet_ind = snippets['indices'][i]
        fig.add_trace(go.Scatter(x=np.arange(snippet_ind, snippet_ind+m), y=list(ts[snippet_ind:snippet_ind+m]), name=f"Snippet {i} in time series", line=dict(color=colors[i+1])), row=1, col=1)

    #plot snippets
    for i in range(snippets_num):
        snippet_ind = snippets['indices'][i]
        fig.add_trace(go.Scatter(x=np.arange(m), y=list(ts[snippet_ind:snippet_ind+m]), name=f"Snippet {i}", line=dict(color=colors[i+1])), row=2, col=1)

    # plot the regimes of snippets
    regimes = snippets['regimes']
    for i in range(snippets_num):
        slices_of_indices = regimes[np.where(regimes[:,0]==i)][:,1:]
        #print(f'indices slice for the regime of the snippets #{i} is: {slices_of_indices}')
        
        indexes_with_values = []
        for per_slice in slices_of_indices:
            start_idx = per_slice[0]
            stop_idx = per_slice[1]#+m-1
            indexes_with_values = indexes_with_values + list(np.arange(start_idx, stop_idx))

        indexes_none = []
        indexes_none = list(set(list(np.arange(n))) - set(indexes_with_values))
        
        ts_regimes = list(ts)
        for j in range(len(indexes_none)):
            ind = indexes_none[j]
            ts_regimes[ind] = None

        fig.add_trace(go.Scatter(x=np.arange(n), y=ts_regimes, line=dict(color=colors[i+1]), name=f"Regimes {i}"), row=3, col=1)

    fig.update_xaxes(showgrid=False,
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=14, color='black'),
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)

    fig.update_yaxes(showgrid=False,
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=14), color='black',
                     zeroline=False,
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)

    fig.update_layout(title_text="Top-k snippets in the time series",
                      title_x=0.5,
                      title_font=dict(size=16, color='black'),
                      legend_tracegroupgap = 60,
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=10, r=10, t=30, b=10),
                      width=1300, height=800,
                      )

    fig.write_image(plot_path, scale=3)


def plot_profiles(profiles, n, snippets_num, plot_path, title = "MPdist-Profiles"):
    
    fig = go.Figure()
    
    for i in range(snippets_num):
        fig.add_trace(go.Scatter(x=list(np.arange(n)),
                                 y=list(profiles[i]),
                                 name=f"MPdist-Profile {i}"))

    fig.update_xaxes(showgrid=False,
                     title='Time',
                     title_font=dict(size=14, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=14, color='black'),
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)

    fig.update_yaxes(showgrid=False,
                 title='Values',
                 title_font=dict(size=14, color='black'),
                 linecolor='#000',
                 ticks="outside",
                 tickfont=dict(size=14, color='black'),
                 zeroline=False,
                 linewidth=1,
                 tickwidth=1,
                 mirror=True)

    fig.update_layout(title=title,
                      title_font=dict(size=16, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=10, r=10, t=40, b=10),
                      width=1200, height=300
                      )

    fig.write_image(plot_path, scale=3)


def bar_plot_profile_area(areas, snippets_num, plot_path):
    """
    Find the optimal k (number of snippets)
    """
    
    x_labels = list(map(str, list(np.arange(1,snippets_num+1))))
    
    #fig = go.Figure([go.Bar(x=animals, y=[20, 14, 23])])
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Profile Area", "Change in Profile Area"))
    fig.add_trace(go.Bar(x=x_labels, y=list(areas)), row=1, col=1)
    fig.add_trace(go.Bar(x=x_labels, y=list(areas[:-1]/areas[1:] - 1.0)), row=2, col=1)
    
    fig.update_layout(title_text="Find the optimal k (number of snippets)") # height=600, width=600,

    fig.write_image(plot_path, scale=3)


def plot_annotation(ts, train_label, predicted_label, n, plot_path, title="Anomalies annotation"):

    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("Time Series", "True labels", "Predicted labels")
                        )

    # plot time series with discords
    fig.add_trace(go.Scatter(x=list(np.arange(n)),
                         y=list(ts),
                         line=dict(color='#636EFA'),
                         name="Time Series"),
              row=1, col=1)

    # plot true labels
    fig.add_trace(go.Scatter(x=list(np.arange(len(train_label))),
                             y=list(train_label),
                             line=dict(color='#00CC96', width=2),
                             name="True labels"),
                  row=2, col=1)
                  
    # plot predicted labels
    fig.add_trace(go.Scatter(x=list(np.arange(len(predicted_label))),
                             y=list(predicted_label),
                             line=dict(color='#EF553B', width=2),
                             name="Predicted labels"),
                row=3, col=1)
                  
    fig.update_annotations(font=dict(size=16, color='black'))

    fig.update_xaxes(showgrid=False,
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=14, color='black'),
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)

    fig.update_yaxes(showgrid=False,
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=14, color='black'),
                     zeroline=False,
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)

    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      showlegend=False,
                      margin=dict(l=10, r=10, t=30, b=10),
                      width=1200, height=800,
                      )
                  
    fig.write_image(plot_path, scale=3)
