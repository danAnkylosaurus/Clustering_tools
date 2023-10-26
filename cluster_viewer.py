#%%
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib as mpl

import functions as fs

#%%

# catalog location

df=fs.read_calif('./data/catalog.csv')


time_const = 24*60*60*30*50

df.time_coors/=time_const
clst = HDBSCAN(min_samples=15, cluster_selection_epsilon=0.05, store_centers='centroid').fit(df.loc[:,['lat','lon','time_coors']])
print(set(clst.labels_))
df['clst_mark'] = clst.labels_

clst_len = len(set(clst.labels_))
col_map = mpl.colormaps['jet']
df['clst_cols'] = [mpl.colors.rgb2hex(col_map((i+2)/(df.clst_mark.max()+2))) for i in df.clst_mark]


#%%
from dash import Dash, html, Input, Output, dash_table, dcc, callback

app = Dash(__name__)

cdict={
    'dd':'#15100A', #dark-dark
    'd':'#362E2D',  #dark
    'hl':'#a84b01', #highLight
    'l':'#AAACA1',  #light
    'll':'#C7C9C6',
    'graph_bgc':'#5b5150'
}

round_stl = {'border-radius':'15px', 'background-color':cdict['d'], 'padding':'25px'}

# fisrt row of graphs

fig_3dgraph = go.Figure( data=[go.Scatter3d(
    x = df.lon,
    y = df.lat,
    z = df.time_coors,
    mode = 'markers',
    marker_colorscale="plasma",
    marker = fs.marker_design_dict(df),
    text = df.clst_mark,
    hovertemplate = """Loc: %{y}, %{x} <br>Class: %{text} <br><extra></extra>"""
    )],
    layout=go.Layout(margin={'r':10,'l':10,'b':10,'t':10}, height=600, 
                     paper_bgcolor='rgba(0,0,0,0)',
                     scene=fs.scene_axis_limit(df.lon, df.lat, df.time_coors, colcol=cdict['ll']))
)

fig_mapbox = go.Figure(go.Scattermapbox(
        lat=df.lat,
        lon=df.lon,
        mode='markers',
        marker_colorscale="plasma",
        marker=fs.marker_design_dict(df),
        text=df.clst_mark,
        hovertemplate="""Loc: %{lat}, %{lon} <br>Class: %{text} <br><extra></extra>"""
        ), layout=fs.map_mapbox_layout(df,hh=600)
        )


# second row of graphs

fig_table = go.Figure( data=[go.Table(
        header=dict(values=fs.table_heading_list(), fill_color=cdict['graph_bgc'],font=dict(color=cdict['ll'])),
        cells=dict(values=fs.table_content_list(df.sort_values('hypoTime',ascending=True)), fill_color=cdict['ll']),
        columnwidth=[2,1,1,1,1,1])
    ],
    layout=fs.layout_margins(10, fcolcol=cdict['d'])
)


# third row

fig_allmag_in_time = go.Figure(go.Scatter(
    x = df.hypoTime, 
    y = df.mag, 
    mode='markers', 
    marker_colorscale="plasma",
    marker = fs.marker_design_dict(df),
    text=df.clst_mark,
    hovertemplate="""Loc: %{x} <br>M%{y}, <br>Class: %{text} <br><extra></extra>"""
    ), layout=fs.layout_margins(5, colcol=cdict['ll'], fcolcol=cdict['ll'])
)


# forth row
fig_multiple_hours = go.Figure(data=[],
    layout=fs.layout_margins(5, colcol=cdict['ll'], fcolcol=cdict['ll'])
)



app.layout = html.Div([
    html.H1( children=['Cluster viewer'] ),
    html.Hr(style={'color':cdict['hl']}),
    html.H3('Select cluster number:'),
    html.H4( dcc.Dropdown(list(set(df.clst_mark)), multi=True, id="input_clst",style={'padding': '10px'}) ),
    html.Div([
        html.H2('Geografical and time coordinates'),
        html.Tbody([
            html.Tr([html.Td( dcc.Graph(figure=fig_3dgraph, id='3d_graph'), style={'width':'50vw','height':600, "padding-bottom": "20px"} ),
                    html.Td( dcc.Graph(figure=fig_mapbox,  id='2d_map'), style={'width':'50vw', "padding-bottom": "20px"} )
                    ]),
            html.Tr([
                    html.Td(dcc.Graph(figure=fig_table, id="clst_table"), colSpan=2, style={"padding-left": "10%", "padding-right": "10%","padding-top": "20px", "border-top":"2px solid %s"%cdict['d']  })
                    ],
                    style={"justify" : "center","padding-left": "100px"})
                    
        ],),
    ], style=round_stl
    ),

    html.Br(),

    html.Div([
        html.H2('Magnitude in time'),
        dcc.Graph(figure=fig_allmag_in_time, id='magnitude_time_graph') 
        ], 
        style=round_stl
    ),

    html.Br(),

    html.Div([
        html.H2('Hourly distribution', ),
        dcc.Graph(figure=fig_multiple_hours, id='table_of_graphs') 
        ], 
        style=round_stl
    )

    ], 
    style={ 'padding':'25px',"outline": "0px"},
)


@callback(
    Output("3d_graph", "figure"),
    Output("2d_map", "figure"),
    Output("clst_table","figure"),
    Output('table_of_graphs', "figure"),
    Output('magnitude_time_graph','figure'),
    Input("input_clst", "value"),
    )
def update_graph(value):
    print(value)

    if value!=None and value!=[]:
        dff = df[df['clst_mark'].isin(value)]
        fig1 = go.Figure( data=[go.Scatter3d( x=dff.lon, y=dff.lat, z=dff.time_coors, mode = 'markers', marker = fs.marker_design_dict(dff))], 
                          layout=go.Layout( margin={'r':10,'l':10,'b':10,'t':10}, paper_bgcolor='rgba(0,0,0,0)', 
                                           scene=fs.scene_axis_limit(df.lon, df.lat, df.time_coors, colcol=cdict['ll'])))
        
        fig2 = go.Figure(go.Scattermapbox(lon=dff.lon, lat=dff.lat, mode='markers', 
                                          marker = fs.marker_design_dict(dff),
                                          text=dff.clst_mark, hovertemplate="""Loc: %{lat}, %{lon} <br>Class: %{text} <br><extra></extra>"""
                                          ), layout=fs.map_mapbox_layout(dff))
        
        fig3 = go.Figure(go.Table(
                header=dict(values=fs.table_heading_list(), fill_color=cdict['graph_bgc'], font=dict(color=cdict['ll'])),
                cells=dict(values=fs.table_content_list(dff.sort_values('mag',ascending=False)),  fill_color=cdict['ll']),
                columnwidth=[2,1,1,1,1,1]
                ),layout=fs.layout_margins(10, colcol=cdict['graph_bgc'], fcolcol=cdict['d'])
                )
        
        fig5 = go.Figure(go.Scatter(
                x = dff.hypoTime, 
                y = dff.mag, 
                mode='markers', 
                marker_colorscale="plasma",
                marker = fs.marker_design_dict(dff),
                text=df.clst_mark,
                hovertemplate="""Loc: %{x} <br>M%{y}, <br>Class: %{text} <br><extra></extra>"""
                ), layout=fs.layout_margins(5, colcol=cdict['ll'], fcolcol=cdict['ll'])
                )
        fig5.update_xaxes(dict(range=[df.hypoTime.min()-pd.Timedelta(days=7), df.hypoTime.max()+pd.Timedelta(days=7)]))

        # part for fig4
        fig4=make_subplots( rows=len(value), cols=2, 
                           specs=[[{'type':'xy'}, {'type':'polar'}] for j in range(len(value))],
                           subplot_titles=([j for k in zip(value, ['' for kk in value]) for j in k]),
                           column_widths=[0.7, 0.3],
                           row_heights=list(np.ones(len(value))/len(value))
                           )
        print(fig4)
        for i in range(len(value)):
            dff = df[df['clst_mark']==value[i]]

            fig4.add_trace(go.Scatter(x=dff.hypoTime, y=dff.mag, mode='markers', 
                                      marker=fs.marker_design_dict(dff)
                                ), row=i+1, col=1
            )

            fig4.add_trace(go.Barpolar(
                r=[sum(dff['hypoTime'].dt.hour==j) for j in range(24)],
                theta=np.arange(0.5,24.5)/24*360,
                marker_color=list(set(df[df['clst_mark']==value[i]]['clst_cols']))[0],
                ),row=i+1, col=2
            )

        fig4.update_layout(fs.layout_margins(20, colcol=cdict['ll'], fcolcol=cdict['ll']))
        fig4.update_layout(dict(height=300*len(value), showlegend=False))
        fig4.update_polars(fs.simple_polar_plot(tcolcol=cdict['dd'], bgcolcol=cdict['ll']))
        fig4.update_xaxes(dict(range=[df.hypoTime.min()-pd.Timedelta(days=7), df.hypoTime.max()+pd.Timedelta(days=7)]))

        return fig1, fig2, fig3, fig4, fig5

    else:
        
        return fig_3dgraph, fig_mapbox, fig_table, fig_multiple_hours, fig_allmag_in_time

    

#  http://127.0.0.1:port
app.run_server(port=1222, debug=True)