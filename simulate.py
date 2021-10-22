#!/usr/bin/env python
"""
Simple 2D gaussian + background simulations.
"""
__author__ = "Alex Drlica-Wagner"
import inspect
import yaml

import numpy as np
import pylab as plt
import scipy.stats
import fitsio
from matplotlib.patches import Ellipse

SIZE = 64
DTYPE = [
    ('id',int),                # object id
    ('x',float),               # x-centroid
    ('y',float),               # y-centroid
    ('nsig',int),              # number of signal photons
    ('nbkg',float),            # mean number of background photons
    ('reff',float),            # azimuthally averaged radius [a * sqrt(1 -ell)]
    ('ell',float),             # ellipticity
    ('phi',float),             # position angle
]

def create_population(size=100, nsig=10000, nbkg=10, 
                      x=SIZE//2, y=SIZE//2, reff=100, ell=0, phi=0):
                      
                      
    """ Create the population parameters

    Parameters
    ----------
    size : number of simulations
    nsig : number of signal photons 
    nbkg : number of background photons
    x,y  : centroid location [pix]
    reff : azimuthally averaged radius [pix]
    ell  : ellipticity
    phi  : position angle [deg]
    """
    # Could probably do something clever with
    #inspect.getargvalues(inspect.currentframe())

    ones = np.array([1,1])
    x *= ones
    y *= ones
    nsig *= ones
    nbkg *= ones
    reff *= ones
    ell *= ones
    phi *= ones

    params = np.recarray(size,dtype=DTYPE)
    params['id']   = np.arange(size)
    params['x']    = np.random.uniform(x[0],x[1],size)
    params['y']    = np.random.uniform(y[0],y[1],size)
    params['nsig'] = np.random.uniform(nsig[0],nsig[1],size)
    params['nbkg'] = np.random.uniform(nbkg[0],nbkg[1],size)    
    params['reff'] = np.random.uniform(reff[0],reff[1],size)
    params['ell']  = np.random.uniform(ell[0],ell[1],size)
    params['phi']  = np.random.uniform(phi[0],phi[1],size)

    return params

def create_image(params):
    """Simulate a 2D gaussian with noise from a set of params"""
    bkg = create_background(params)
    sig = create_galaxy(params)
    img = bkg + sig

    return img

def create_galaxy(params):
    """Create the galaxy from a 2D Gaussian distribution"""
    nsig = params['nsig']
    mean = [params['x'],params['y']]
    reff = params['reff']
    ell  = params['ell']
    phi  = params['phi']

    a = reff / np.sqrt(1 - ell)
    b = reff**2 / a

    # https://stackoverflow.com/a/54442206/4075339
    cov_xx = a**2 * np.cos(np.radians(phi))**2 + b**2 * np.sin(np.radians(phi))**2
    cov_yy = a**2 * np.sin(np.radians(phi))**2 + b**2 * np.cos(np.radians(phi))**2
    cov_xy = (a**2 - b**2) * np.sin(np.radians(phi)) * np.cos(np.radians(phi)) 

    cov = np.array([[cov_xx,cov_xy],[cov_xy,cov_yy]])

    # List of photons
    sig = np.random.multivariate_normal(mean,cov,size=nsig)
    cut = np.any((sig < 0) | (sig > SIZE), axis=1)
    if np.any(cut == True):
        print(f"WARNING: {cut.sum()} photons outside of image")
        sig = sig[~cut]
        
    img = np.zeros(shape=(SIZE,SIZE))
    xidx = sig[:,0].astype(int)
    yidx = sig[:,1].astype(int)
    np.add.at(img, (yidx,xidx), 1)
    return img

def create_background(params):
    """ Create uniform background from poisson distribution """
    nbkg = params['nbkg']
    img = scipy.stats.poisson.rvs(nbkg, size=(SIZE,SIZE))
    return img

def create_ellipse(params, **kwargs):
    """ Create matplotlib.patches.Ellipse """
    mean = [params['x'],params['y']]
    reff = params['reff']
    ell  = params['ell']
    phi  = params['phi']

    a = reff / np.sqrt(1 - ell)
    b = reff**2 / a
    
    return Ellipse(mean, 2*a, 2*b, angle=phi, **kwargs)

def draw_ellipse(params, **kwargs):
    """ Draw the matplotlib ellipse """
    ellipse = create_ellipse(params,**kwargs)
    ax = plt.gca()
    ax.add_artist(ellipse)
    return ax, ellipse

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',
                        help='configuration file')
    parser.add_argument('-n','--nsims',default=100,type=int,
                        help='number of galaxies to simulate')
    parser.add_argument('-p','--plot',action='store_true',
                        help='plot simulated images')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    
    # Example: Create a population with fixed parameters
    #catalog = create_population(size=args.nsims,reff=10,ell=0.5,phi=45)

    # Example: Create a population sampling from random uniform distributions of sizes
    #catalog = create_population(size=args.nsims,nsig=[1000,10000],reff=[5,20])

    # Example: Create a population from the config file
    catalog = create_population(size=args.nsims,nsig=config['nsig'],reff=config['reff'])

    # Save to npy file
    np.save('catalog.npy',catalog)

    # Generate galaxy images
    image_array = []
    for i,params in enumerate(catalog):
        img = create_image(params)
        image_array.append(img)
        if args.plot:
            plt.clf()
            plt.imshow(img)
            draw_ellipse(params,fill=False,color='r',zorder=1)

    np.save('images.npy',np.array(image_array))

    if args.plot: plt.ion()
        
