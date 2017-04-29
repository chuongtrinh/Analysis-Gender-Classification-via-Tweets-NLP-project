import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import plotly 

plotly.tools.set_credentials_file(username='ctrinh', api_key='Poy9KdV3HQN2p4Liznoe')


def plot(data,labels, should_ignore_brand,file_name):
    male_indices = [i for i,x in enumerate(labels) if x == 'male']
    female_indices = [i for i,x in enumerate(labels) if x == 'female']
    if should_ignore_brand == False:
        brand_indices = [i for i,x in enumerate(labels) if x == 'brand']
    
    trace0 = go.Scatter3d(x = data[male_indices,0],y = data[male_indices,1],z =data[male_indices,2],mode = 'markers',name = 'male',marker = dict(color = 'red', symbol = 'circle',size=3))
    trace1 = go.Scatter3d(x = data[female_indices,0],y = data[female_indices,1],z =data[female_indices,2], mode = 'markers',name = 'female',marker = dict(color = 'blue', symbol = 'circle',size =3))
    if should_ignore_brand == False:
        trace2 = go.Scatter3d(x = data[brand_indices,0],y = data[brand_indices,1],z =data[brand_indices,2],mode = 'markers', name = 'brand',marker = dict(color = 'black', symbol = 'circle',size = 3))
    if should_ignore_brand == False:
        data = [trace0, trace1, trace2]
    else:
        data = [trace0, trace1]
        
    layout = go.Layout(margin=dict(l=0,r=0,b=0,t=0))
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename=file_name)

