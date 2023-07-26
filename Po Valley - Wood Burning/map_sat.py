import matplotlib.pyplot as plt
#import matplotlib.patheffects as PathEffects
from owslib.wmts import WebMapTileService
#import cartopy
import cartopy.crs as ccrs
import pandas as pd


def main():
    # URL of NASA GIBS
    URL = 'http://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi'
    wmts = WebMapTileService(URL)

    
    layers = ['MODIS_Terra_SurfaceReflectance_Bands143']#,
              #'MODIS_Terra_CorrectedReflectance_Bands367']

    date_str = '2022-05-11'#'2016-02-05'
    #date_str = '2022-12-02'
    
    # Plot setu
    plot_CRS = ccrs.Mercator()
    geodetic_CRS = ccrs.Geodetic()
    x0, y0 = plot_CRS.transform_point(6.6, 43.1, geodetic_CRS) #po valley
    x1, y1 = plot_CRS.transform_point(14.0, 47.4, geodetic_CRS) #po valley
    #x0, y0 = plot_CRS.transform_point(65, 15.0, geodetic_CRS)
    #x1, y1 = plot_CRS.transform_point(101.0, 42, geodetic_CRS)
    ysize = 8
    xsize = 2 * ysize * (x1 - x0) / (y1 - y0)
    fig = plt.figure(figsize=(xsize, ysize), dpi=100)

    for layer, offset in zip(layers, [0, 0.5]):
        #ax = fig.add_axes([offset, 0, 0.5, 1], projection=plot_CRS)
        ax = plt.axes(projection=plot_CRS)
        ax.set_xlim((x0, x1))
        ax.set_ylim((y0, y1))
        ax.add_wmts(wmts, layer, wmts_kwargs={'time': date_str})
        gl = ax.gridlines(draw_labels=True,alpha=0.3, dms=False, x_inline=False, y_inline=False);
        gl.ylabel_style = {'size': 15}
        gl.xlabel_style = {'rotation': 0, 'size': 15};
        fig.savefig('C:/Users/manue/Desktop/MAPPA_NEPAL.png', dpi=500)
    plt.show()


if __name__ == '__main__':
    main()
