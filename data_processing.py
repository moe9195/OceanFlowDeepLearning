'''
This script processes the satellite data.
Data is converted to numpy arrays, missing values are filled
Kinetic Energy and Vorticity are calculated from the data
'''

from ocean import *

filenames = ['data/0.nc', 'data/1.nc', 'data/2.nc',
             'data/3.nc', 'data/4.nc', 'data/5.nc',
             'data/6.nc', 'data/7.nc', 'data/8.nc',
             'data/9.nc', 'data/10.nc', 'data/11.nc',
             'data/12.nc', 'data/13.nc', 'data/14.nc',
             'data/15.nc', 'data/16.nc', 'data/17.nc',
             'data/18.nc', 'data/19.nc', 'data/20.nc',
             'data/21.nc', 'data/22.nc', 'data/23.nc']

data = get_data('data/1.nc', 1)
lons, lats, vx, vy, T, time = data

c1, c2, c3 = 0, 0, 0
for fname in filenames:
    data = get_data(fname, 1)
    lons, lats, vx, vy, T, time = data
    KE = (vx**2 + vy**2)
    vor = vorticity(data)
    for i in range(len(time)):
        KE_split = split_matrix(KE[i], 60, 60)
        vor_split = split_matrix(vor[i], 60, 60)
        T_split = split_matrix(T[i], 60, 60)

        frac = np.sum(np.sum(np.isnan(KE_split), 1), 1)/(60*60)
        idx = np.argwhere(frac <= 0.02)

        vv_KE = np.array(KE_split)
        vn_KE = vv_KE[np.ravel(idx)]

        vv_vor = np.array(vor_split)
        vn_vor = vv_vor[np.ravel(idx)]

        vv_T = np.array(T_split)
        vn_T = vv_T[np.ravel(idx)]

        for j in range(len(vn_KE)):
            np.save('./numpy_arrays/kinetic energy/'+str(c1)+'-'+str(c2)+'-'+str(c3), vn_KE[j])
            np.save('./numpy_arrays/vorticity/'+str(c1)+'-'+str(c2)+'-'+str(c3), vn_vor[j])
            np.save('./numpy_arrays/temperature/'+str(c1)+'-'+str(c2)+'-'+str(c3), vn_T[j])
            c3 += 1
        c3 = 0
        c2 += 1
    c2 = 0
    c1 += 1

# impute the missing values from each 120x120 matrix
# and save the imputed matrices into different folders

path1 = './numpy_arrays/'
path2 = './filled_arrays/'
fnames = os.listdir(path1+'temperature/')

for fname in fnames:
    tmp1 = np.load(path1+'temperature/'+fname)
    tmp2 = np.load(path1+'kinetic energy/'+fname)
    tmp3 = np.load(path1+'vorticity/'+fname)

    # better imputation at significant computational cost increase
    #filled1 = fill_nans_iterative(tmp1,8)
    #filled2 = fill_nans_iterative(tmp2,8)
    #filled3 = fill_nans_iterative(tmp3,8)

    filled1 = fill_nans_simple(tmp1)
    filled2 = fill_nans_simple(tmp2)
    filled3 = fill_nans_simple(tmp3)

    np.save(path2+'temperature/'+fname, filled1)
    np.save(path2+'kinetic energy/'+fname, filled2)
    np.save(path2+'vorticity/'+fname, filled3)

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

# setup directory structure for training

filelist = os.listdir('./filled_arrays/vorticity/')
count = 0
for fname in filelist:
    if count > (len(filelist) - 40000):
        shutil.move('./filled_arrays/temperature/'+fname,
                    './dataset/temperature/validate/X/'+fname)
        shutil.move('./filled_arrays/vorticity/'+fname,
                    './dataset/temperature/validate/Y/'+fname)
        shutil.move('./filled_arrays/kinetic energy/'+fname,
                    './dataset/kinetic energy/validate/X/'+fname)
    else:
        shutil.move('./filled_arrays/temperature/'+fname,
                    './dataset/temperature/train/X/'+fname)
        shutil.move('./filled_arrays/vorticity/'+fname,
                    './dataset/temperature/train/Y/'+fname)
        shutil.move('./filled_arrays/kinetic energy/'+fname,
                    './dataset/kinetic energy/train/X/'+fname)
    count += 1
