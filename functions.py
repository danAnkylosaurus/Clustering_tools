#%%
import pandas as pd
import datetime
import numpy as np
import matplotlib as mpl


# read functions that reshape input files for the programm

def read_calif(flnm):
    df = pd.read_csv(flnm, delimiter=' ', skiprows=9, skipinitialspace=True)
    df['hypoTime'] = [datetime.datetime.strptime(df['#YYY/MM/DD'][i]+' '+df['HH:mm:SS.ss'][i],'%Y/%m/%d %H:%M:%S.%f') for i in range(len(df))]
    df['time_coors'] = [df['hypoTime'][i].timestamp() for i in range(len(df))]
    
    df = df.loc[:,['MAG','LAT','LON','DEPTH','hypoTime','time_coors']]
    df = df.rename(columns={'MAG':'mag', 'LAT':'lat','LON':'lon', 'DEPTH':'depth'})

    return df


# functions for styling graphs

def table_heading_list():
    return ['Time', 'lat', 'lon', 'mag', 'depth', 'clst_mark']

def table_content_list(datf):
    return [datf.hypoTime,  datf.lat, datf.lon, datf.mag, datf.depth, datf.clst_mark]

def marker_design_dict(datf,opac=0.7,colcolmap=mpl.colormaps['plasma']):
    return dict(
        size=datf.mag**2+5,
        color=datf.clst_cols,              
        opacity=opac,
    )

def scene_axis_limit(xx,yy,zz,pp=0,colcol="#ffffff"):
    return dict(
                xaxis = dict(range=[min(xx)*(1-pp),max(xx)*(1+pp)], color=colcol, title='lon'),
                yaxis = dict(range=[yy.min()*(1-pp),yy.max()*(1+pp)], color=colcol, title='lat'),
                zaxis = dict(range=[zz.min()*(1-pp),zz.max()*(1+pp)], color=colcol, title='time'),)

import plotly.graph_objects as go

def map_mapbox_layout(datf,hh=600):
    return go.Layout(
        mapbox_style="open-street-map",
        mapbox_bounds={"west": datf.lon.min()-2, "east": datf.lon.max()+2, "south":datf.lat.min()-2, "north": datf.lat.max()+2},
        height=hh,
        margin={'r':10,'l':10,'b':10,'t':10},
        paper_bgcolor='rgba(0,0,0,0)'
        )

def layout_margins(nn,colcol='white', fcolcol='blue'):
    return go.Layout( margin={'r':nn,'l':nn,'b':nn,'t':nn}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=colcol, font_color=fcolcol)

def simple_polar_plot(tcolcol='black', bgcolcol='white'):
    return dict(
                radialaxis = dict(showticklabels=True, color=tcolcol, tickangle=45),
                angularaxis = dict(direction='clockwise',rotation=90,
                                    dtick=360/24,
                                    tick0=0,
                                    tickmode='array',
                                    ticktext=np.arange(0,24),
                                    tickvals=np.arange(0,24)*15),
                bgcolor=bgcolcol)


