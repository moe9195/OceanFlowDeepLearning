'''
A variety of helper functions needed for this project
'''

# importing relevant packages
from sklearn.experimental import enable_iterative_imputer
from tensorflow.keras.models import load_model
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn import model_selection
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import xarray as xr
import pickle, csv, fileinput, shutil, os, re
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# takes a vector of x-locations and a vector of y-locations
# and calculates the slope (dy/dx) at each x-location.
def slope(y, x):
    l = np.size(y)
    s = np.zeros(l)
    s[0] = (y[1] - y[0]) / (x[1] - x[0])
    s[l - 1] = (y[l - 1] - y[l - 2]) / (x[l - 1] - x[l - 2])
    for i in range(1, l - 1, 1):
        s[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
    return s

# calculates curl of a vector F on a 2D rectangular grid
def curl(x, y, Fx, Fy):
    dFy_dx = np.zeros((len(y), len(x)))
    dFx_dy = np.zeros((len(y), len(x)))

    for iy in range(len(y)):
        dFy_dx[iy, :] = slope(np.ravel(Fy[iy, :]), x)

    for ix in range(len(x)):
        dFx_dy[:, ix] = slope(np.ravel(Fx[:, ix]), y)

    return dFy_dx - dFx_dy

# calculates the vorticity when given the ocean data from the get_data function
def vorticity(data):
    lons, lats, vx, vy = data[0], data[1], data[2], data[3]
    w, h, t_total = np.shape(vx)[1], np.shape(vx)[2], np.shape(vx)[0]
    vor = np.zeros((t_total, w, h), dtype=np.float32)
    for i in range(t_total):
        vor[i, :, :] = curl(lons, lats, vx[i, :], vy[i, :])
    return vor

# takes the .nc filename and outputs its data as numpy arrays
def get_data(filename, n):
    ds = xr.open_dataset(filename)

    d, time = ds.depth, ds.time
    lat, long = ds.latitude, ds.longitude

    if n == 1:
        v_x, v_y = ds.uo, ds.vo
        temp = ds.thetao
    else:
        v_x, v_y = ds.u, ds.v
        temp = ds.temperature

    temp = temp.values
    lats, lons = lat.values, long.values
    vx, vy = v_x.values, v_y.values

    w = np.shape(vx[0, :])[1]
    h = np.shape(vx[0, :])[2]
    t_total = len(time.values)
    w, h = np.shape(vx[0, :])[1], np.shape(vx[0, :])[2]
    vx, vy, temp = vx.reshape((t_total, w, h)), vy.reshape((t_total, w, h)), temp.reshape((t_total, w, h))
    data = [lons, lats, vx, vy, temp, time]
    return data

# takes a 3d matrix and splits each submatrix into 4 submatricies for each iteration
# input is an NxMxM matrix and output is N'xM'xM' matrix
# where N' = (4^iterations)*N and M' = M/(2^iterations)
def split_data(A, iterations):
    def split_in_four(A):
        shape = np.shape(A)
        d, l, w = shape[0], shape[1], shape[2]
        div = int(l/2)
        upperLeft  = A[0:d,0:div,0:div]
        upperRight = A[0:d,0:div,div:2*div]
        lowerLeft  = A[0:d,div:2*div,0:div]
        lowerRight = A[0:d,div:2*div,div:2*div]
        combined   = np.concatenate((upperLeft, upperRight, lowerLeft, lowerRight))
        return combined
    for i in range(iterations):
        A = split_in_four(A)
        if np.shape(A)[2] < 2:
            break
    return A

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def fill_nans_simple(A):
    if len(A.shape)>2:
        A0 = np.empty(np.shape(A))
        for i in range(len(A)):
            imp = SimpleImputer()
            imp.fit(A[i, :])
            A0[i, :] = imp.transform(A[i, :])
        return A0
    else:
        nans, x= nan_helper(A)
        A[nans]= np.interp(x(nans), x(~nans), A[~nans])
        return A

# fills NaN values by modelling each feature as a function of missing futures
# gives better results than simple imputer but much more computationally heavy
def fill_nans_iterative(A, n):
    imp_mean = IterativeImputer(random_state=0, n_nearest_features=n)
    imp_mean.fit(A)
    A0 = imp_mean.transform(A)
    return A0


# takes an n-dimensional array and normalizes its values to [0, 1]
def normalize(x):
    return (x - np.nanmin(x))/(np.nanmax(x) - np.nanmin(x))

# function for plotting the data. set plot_quiver = 1 to plot vector field,
# set plot_vorticity, plot_temperature, or plot_KE to 1 and the others to 0 to
# plot the desired variable
def plot_data(t, data, plot_quiver,
              plot_vorticity, plot_temperature, plot_KE):

    skip = 5 # skips data points when plotting vector field for better visualisation
    lons, lats, vx, vy, time, T = data[0], data[1], data[2], data[3], data[4], data[5]

    w, h = np.shape(vx)[1], np.shape(vx)[2]
    vx_s, vy_s = vx[t,:].reshape((w,h)), vy[t,:].reshape((w,h))
    time = time[t]

    if plot_vorticity:
        if not plot_temperature:
            vor = curl(lons, lats, vx_s, vy_s)   # calculating vorticity
           # vmin = np.nanmin(vor)/2
           # vmax = np.nanmax(vor)/2
            vmin = -5.5
            vmax = 5.5
            label = 'Vorticity'
        else:
            vor = T[t,:]
            vmin = np.nanmin(vor)
            vmax = np.nanmax(vor)
            label = 'Temperature'
    if plot_KE:
        vor = np.sqrt(vx_s**2 + vy_s**2)/2
        vmin = np.nanmin(vor)
        vmax = np.nanmax(vor)
        label = 'Kinetic Energy'

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(
        central_longitude=0.0, globe=None)})
    fig.set_size_inches([30,15])

    ax.set_global()
    ax.stock_img()
    ax.text(0.55, 0.95, str(time)[0:10],
        transform=ax.transAxes, ha="right", color='black',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    if plot_vorticity:
        plt.contourf(lons, lats, vor, 99, levels=np.linspace(-5.5, 5.5, 100, endpoint=True),
                     vmin = vmin, vmax = vmax, transform=ccrs.PlateCarree())

        cbaxes = inset_axes(ax, width="3.5%", height="45%", loc=2)
        cbar = plt.colorbar(cax=cbaxes, ticks=np.arange(-5,6,1));
        cbar.ax.set_ylabel(label, rotation=270, labelpad=10)

    if plot_quiver:
        ax.quiver(lons[::skip], lats[::skip], vx_s[::skip, ::skip], vy_s[::skip, ::skip],
            transform=ccrs.PlateCarree(), width=0.002)

    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())
    plt.savefig('Images/' + 'img-'+str(t)+'.jpeg', bbox_inches = 'tight', pad_inches = 0, dpi=100)
    plt.clf()

def generate_data(directory, X, batch_size):
    i = 0
    file_list = np.array(sorted(os.listdir(directory+'/X'), key=numericalSort))
    while True:
        array_batch1 = []
        array_batch2 = []
        ind = 0
        for i in range(batch_size):
            if ind == len(file_list):
                ind = 0
                np.random.shuffle(file_list)
            sample = file_list[ind]
            ind+=1
            array1 = np.load(directory+ X +sample)
            array2 = np.load(directory+'/vorticity/'+sample)
            array_batch1.append(array1)
            array_batch2.append(array2)
        yield (np.array(array_batch1).reshape(batch_size, 120, 120, 1),
               np.array(array_batch2).reshape(batch_size, 120, 120, 1))

def test_model(directory, batch_size):
    i = 0
    file_list = np.array(sorted(os.listdir(directory+'/X'), key=numericalSort))
    while True:
        array_batch1 = []
        array_batch2 = []
        ind = 0
        for i in range(batch_size):
            if ind == len(file_list):
                ind = 0
                np.random.shuffle(file_list)
            sample = file_list[ind]
            ind+=1
            array1 = np.load(directory+'/X/'+sample)
            array2 = np.load(directory+'/Y/'+sample)
            array_batch1.append(array1)
            array_batch2.append(array2)
        yield (np.array(array_batch1).reshape(batch_size, 75, 75, 1),
               np.array(array_batch2).reshape(batch_size, 75, 75, 1))

def split_matrix(A, n, m):
    x, y = A.shape
    A_split = []
    idx = 0
    for i in range(0, x-1, n):
        for j in range(0, y-1, m):
            tmp = A[i:(i+n), j:(j+m)]
            A_split.append(tmp)
    return A_split

def numericalSort(list):
    def key(value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    return sorted(list, key=key)

# Data Visualisation
splitt = split_matrix(T[0], 60,60)
tst = splitt[145]
idx = 0
c=0
n, m = 0, 0
joind = np.empty((2041, 4320))
frac = np.sum(np.sum(np.isnan(splitt), 1), 1)/(60*60)
idd = np.argwhere(frac < 0.02)
ss = np.array(splitt)
ss[np.ravel(idd)] = -1000*np.ones((60,60))

for i in range(2041//60):
    for j in range(4320//60):
        joind[n*60:(n+1)*60, m*60:(m+1)*60] = ss[idx]
        if (2*108<idx<2*114 or 2*144<idx<2*150 or 2*180<idx<2*186 or 2*216<idx<2*222
           or 2*252<idx<2*258 or 2*288<idx<2*294 or 2*324<idx<2*330):

            joind[n*60:(n+1)*60, m*60:(m+1)*60] = 1000*np.ones((60,60))
            c+=1
        m+=1
        idx+=1
    n+=1
    m=0

plt.imshow(joind, origin='lower')

# setting up the test regions
idd = np.ravel(idd)
x, y = T.shape[1], T.shape[2]
globe = np.empty((x, y))
globe[:] = np.nan
a, b, c = 1388, 14, 23

#n = b*c - 1
n = 1
s = fnames[a*(n-1)+n-1:n*(a+1)]
ii=0
idx = 0
chk=0
n, m = 0, 0
south_pacific=[]
for i in range(x//60):
    for j in range(y//60):
        if chk == idd[ii]:
            globe[n*60:(n+1)*60, m*60:(m+1)*60] = np.load(
                './filled_arrays/temperature/'+s[ii])
            ii+=1
        if (2*108<idx<2*114 or 2*144<idx<2*150 or 2*180<idx<2*186 or 2*216<idx<2*222
           or 2*252<idx<2*258 or 2*288<idx<2*294 or 2*324<idx<2*330):
            globe[n*60:(n+1)*60, m*60:(m+1)*60] = 1000*np.ones((60,60))
            south_pacific.append(ii)
        chk+=1
        m+=1
        idx+=1
    n+=1
    m=0

plt.imshow(globe, origin='lower')

a, b, c = 1388, 14, 24

for i in range(1, b*c-1):
    s = fnames[a*(i-1)+i-1:i*(a+1)]
    pacific_files = np.array(s)[south_pacific]
    for fname in pacific_files:
        shutil.move('./filled_arrays/kinetic energy/'+fname,
                    './filled_arrays/kinetic energy (testing)/'+fname)
        shutil.move('./filled_arrays/vorticity/'+fname,
                    './filled_arrays/vorticity (testing)/'+fname)
        shutil.move('./filled_arrays/temperature/'+fname,
                    './filled_arrays/temperature (testing)/'+fname)
