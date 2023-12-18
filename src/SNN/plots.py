# coding: utf-8

import numpy as np
import seaborn as sns
import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go


def plot_score_probability(df, plot_path):
    
    score_probability = sns.displot(
                data=df,
                x='min_similarity_score',
                hue='real_label',
                fill=True,
                stat="probability"
                )
                
    score_probability.savefig(plot_path)


def plot_similarity_scores(ts_test, similarity_scores, true_label, N, plot_path):
    
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("Test time series", "Ground truth label", "Anomaly score")
                        )
    
    # plot the test time series
    fig.add_trace(go.Scatter(x=np.arange(N),
                             y=list(ts_test),
                             line=dict(color='#636EFA', width=2),
                             name="Test time series"),
                  row=1, col=1)
                  
    # plot the true labels
    fig.add_trace(go.Scatter(x=np.arange(N),
                             y=list(true_label),
                             line=dict(color='#00CC96', width=2),
                             name="Ground truth label"),
                  row=2, col=1)
                  
    #plot the predicted scores
    fig.add_trace(go.Scatter(x=np.arange(N),
                             y=list(similarity_scores),
                             line=dict(color='#EF553B', width=2),
                             name="Anomaly score"),
                  row=3, col=1)
    
    fig.update_annotations(font=dict(size=16, color='black'))
    
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
     
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      showlegend=False,
                      margin=dict(l=10, r=10, t=30, b=10),
                      width=1300, height=800,
                      )
    
    fig.write_image(plot_path, scale=3)


def plot_anomaly_regions(ts_test, subs_similarity_scores, anomaly_regions, threshold, N, plot_path):
    
    fig = make_subplots(shared_xaxes=True, rows=2, cols=1)
    
    fig.add_trace(go.Scatter(x=np.arange(N), y=list(ts_test), name="Test Time Series"), row=1, col=1)
    
    for i in range(len(anomaly_regions)):
        fig.add_trace(go.Scatter(x=np.arange(anomaly_regions[i][0], anomaly_regions[i][1]), y=list(ts_test[anomaly_regions[i][0]:anomaly_regions[i][1]]), line=dict(color='red'), showlegend=False), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=np.arange(N), y=list(subs_similarity_scores), line=dict(color='green'), name="Anomaly score"), row=2, col=1)
    fig.add_hrect(y0=threshold, y1=np.max(subs_similarity_scores), line_width=0, fillcolor="red", opacity=0.2, row=2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(N), y=[threshold]*N, line_width=3, line_dash="dash", line=dict(color='red'), name="Threshold"), row=2, col=1)

    fig.write_image(plot_path, scale=3)


def plot_discords(ts, min_score, true_anomaly, predict_anomaly, N, plot_path):
    
    fig = make_subplots(rows=1, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)
        
    # plot time series with discords
    true_anomaly_ts = [ts[anomaly_ind] for anomaly_ind in true_anomaly]
    predict_anomaly_ts = [ts[anomaly_ind] for anomaly_ind in predict_anomaly]
    
    fig.add_trace(go.Scatter(x=np.arange(N), y=list(ts), line=dict(color='#636EFA'), name="Time Series"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(true_anomaly), y=true_anomaly_ts, mode='markers', marker=dict(symbol='star', color='green', size=7), name="True Anomalies"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(predict_anomaly), y=predict_anomaly_ts, mode='markers', marker=dict(symbol='star', color='red', size=7), name="Predict Anomalies"), row=1, col=1)
    
    fig.update_layout(title_text="Top-k discords in the time series")

    fig.write_image(plot_path, scale=3)
