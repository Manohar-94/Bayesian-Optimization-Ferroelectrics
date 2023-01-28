import os, glob
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from adjustText import adjust_text
import plotly.graph_objects as go

def prettyplot():
    """
    prettyplot()——some aesthetically pleasing settings. 
    """
    plt.style.use('seaborn')
    mpl.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['figure.dpi'] = 200

def vis_pund(pund_data, iv_data, device, sheet_name):
    plt.clf()
    plt.plot(pund_data[:,0], pund_data[:,1], c = '#580F41', label = "PUND")
    plt.plot(iv_data[:,0], iv_data[:,1], c = '#A9561E', label = "Current")
    plt.xlabel('time (s)')
    plt.ylabel('V (V)')
    dir = "/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/plots"
    if sheet_name == 'Data':
        iter = '3 cycle'
    elif sheet_name == 'Append1':
        iter = '10^6 cycle'
    elif sheet_name == 'Append2':
        iter = '10^7 cycle'
    plt.title(device + " PUND "+ iter)
    plt.legend(loc="upper right")
    plt.savefig(dir + "/PUND/" + device + "_" + iter +"_PUND-plot")   

def vis_pv(pv_data, pv_pos_tup, pv_neg_tup, device):
    """
    vis_pv(pv_data, pv_pos_tup, pv_neg_tup, device) saves a plot of 
    the shifted PV curve, as well as labelled max and min points.
    """
    plt.clf()
    plt.plot(pv_data[:,0], pv_data[:,1], c = '#580F41', label = "raw")
    plt.plot(pv_pos_tup[0], pv_pos_tup[1], 'C3o')
    plt.plot(pv_neg_tup[0], pv_neg_tup[1], 'C3o')
    plt.xlabel('Vforce (V)')
    plt.ylabel('Charge Density (mC/cm^2)')
    together = [pv_pos_tup, pv_neg_tup]
    texts=[]
    for i_x, i_y in together:
        texts.append(plt.text(i_x, i_y, r'({0:.3f}, {1:.3E})'.format(i_x, i_y), 
        c="black", fontsize='medium', 
        bbox=dict(facecolor='white', alpha=0.5)))
    dir = "/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/plots"
    plt.title(device + " PV")
    plt.legend(loc="upper right")
    adjust_text(texts, only_move={'points':'y', 'texts':'y'})
    plt.savefig(dir + "/PV/" + device + "_" + "PV-plot")   
    # plt.show() 

def vis_iv(iv_data, iv_filt, pos_tup, neg_tup, device):
    """
    vis_iv(iv_data, iv_filt, pos_tup, neg_tup, device) saves a plot of 
    the IV curve, with raw and filtered data, as well as labelled max and min 
    points.
    """
    plt.clf()
    plt.plot(iv_data[:,0], iv_data[:,1], c = '#580F41', label = "raw")
    plt.plot(iv_data[:,0], iv_filt, c = '#A9561E', label = "filtered")
    plt.plot(pos_tup[0], pos_tup[1], 'C3o')
    plt.plot(neg_tup[0], neg_tup[1], 'C3o')
    plt.xlabel('Vforce (V)')
    plt.ylabel('Imeas (mC)')
    col = [pos_tup, neg_tup]
    for i_x, i_y in col:
        plt.text(i_x, i_y, r'({0:.3f}, {1:.3E})'.format(i_x, i_y), 
        c="black", fontsize='medium', 
        bbox=dict(facecolor='white', alpha=0.5))
    dir = "/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/plots"
    plt.title(device + " IV")
    plt.legend(loc="upper right")
    plt.savefig(dir + "/IV/" + device + "_" + "IV-plot")    
    # plt.show()

def vis_pred(noise, column_mean, column_sd, train_x, train_y, test_grid, obs, nshape):
    pred_labels = obs.mean.view(nshape)

    # Get back real values from standardized version
    x_raw = lambda x_stand, sd, x_mean : x_stand*sd + x_mean
    sd_0, sd_1 = column_sd[0], column_sd[1]
    mu_0, mu_1 = column_mean[0], column_mean[1]

    fig = go.Figure(data=[go.Surface(z=pred_labels.numpy().T, 
                                    x=x_raw(test_grid[:,0],sd_0, mu_0),
                                    y=x_raw(test_grid[:,1],sd_1, mu_1),
                                    name='GP regression')])
    fig.add_trace(go.Scatter3d(x=x_raw(train_x[:,0].numpy(), sd_0, mu_0),
                              y=x_raw(train_x[:,1].numpy(), sd_1, mu_1),
                            z=train_y.numpy(), mode='markers', 
                            marker={'color':'darkgreen'}, name='training data'))
    fig.update_layout( width=1000, height=800,
                    legend=dict(orientation="h", yanchor="top", y=1.02, 
                    xanchor="left",x=1), margin=dict(r=20, l=10, b=10, t=10), 
                        scene=dict(
                        xaxis_title="Voltage (V)",
                        yaxis_title="Pulse Width (msec)",
                        zaxis_title='2 Qsw/(U+|D|)')
                    )
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-2, y=-2.5, z=1.75)
    )

    fig.update_layout(scene_camera=camera)
    dir = "/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/plots"
    # plt.savefig(dir + "/noise= "+ str(noise) + ", fig1.png")   
    fig.show()

def vis_acq(noise, column_mean, column_sd, train_x, train_y, test_grid, pred_labels, 
            upper_surf, lower_surf, ucb, th, pi, ei, ca):
    # Get back real values from standardized version
    x_raw = lambda x_stand, sd, x_mean : x_stand*sd + x_mean
    sd_0, sd_1 = column_sd[0], column_sd[1]
    mu_0, mu_1 = column_mean[0], column_mean[1]

    fig = go.Figure(data=[go.Surface(z = upper_surf, x=x_raw(test_grid[:,0], sd_0, mu_0),
                    y=x_raw(test_grid[:,1], sd_1, mu_1), opacity=0.5, showscale=False)])
    fig.add_trace(go.Surface(z=lower_surf, x=x_raw(test_grid[:,0], sd_0, mu_0), 
                            y=x_raw(test_grid[:,1], sd_1, mu_1), 
                            opacity=0.2, showscale=False))
    fig.add_trace(go.Scatter3d(x=x_raw(train_x[:,0].numpy(), sd_0, mu_0),
                                y=x_raw(train_x[:,1].numpy(), sd_1, mu_1), 
                                z=train_y.numpy(), 
                                mode='markers', 
                                name='training data', 
                                marker={'color':'darkgreen'}))

    fig.add_trace(go.Scatter3d(x=[x_raw(test_grid[ucb[1], 0].numpy(), sd_0, mu_0)], 
                                y=[x_raw(test_grid[ucb[1], 1].numpy(), sd_1, mu_1)],
                                z=[pred_labels[ucb[0], ucb[1]]], mode='markers', 
                                name='max(upper confidence bound)')) 

    fig.add_trace(go.Scatter3d(x=[x_raw(test_grid[th[1], 0].numpy(), sd_0, mu_0)], 
                                y=[x_raw(test_grid[th[0],1].numpy(), sd_1, mu_1)],
                                z=[pred_labels[th[0], th[1]].detach().numpy()],
                                 mode='markers', name='max(thompson)')) 
    fig.add_trace(go.Scatter3d(x=[x_raw(test_grid[pi[1], 0].numpy(), sd_0, mu_0)], 
                                y=[x_raw(test_grid[pi[0], 1].numpy(), sd_1, mu_1)],
                                z=[pred_labels[pi[0], pi[1]]], 
                                mode='markers', name='max(pi)'))

    fig.add_trace(go.Scatter3d(x=[x_raw(test_grid[ei[1], 0].numpy(), sd_0, mu_0)], 
                            y=[x_raw(test_grid[ei[0], 1].numpy(), sd_1, mu_1)],
                            z=[pred_labels[ei[0], ei[1]]], mode='markers', 
                            name='max(ei)'))

    fig.add_trace(go.Scatter3d(x=[x_raw(test_grid[ca[1], 0].numpy(), sd_0, mu_0)], 
                                y=[x_raw(test_grid[ca[0], 1].numpy(), sd_1, mu_1)],
                                z=[pred_labels[ca[0], ca[1]]], 
                                mode='markers', name='max(ca)'))

    fig.update_layout( width=800, height=600,
                    margin=dict(r=20, l=10, b=10, t=10),
                    legend=dict(orientation="h", yanchor="bottom", 
                                y=1.02, xanchor="right",x=1),
                    scene=dict(
                        xaxis_title="Voltage (V)",
                        yaxis_title="Pulse Width (msec)",
                        zaxis_title='2 Qsw/(U+|D|)')
                    )
    dir = "/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/plots"
    # plt.savefig(dir + "/noise= " + str(noise) + ", fig2.png")   
    fig.show()