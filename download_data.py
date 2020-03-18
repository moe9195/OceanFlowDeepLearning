'''
This script automates downloading satellite data from CMEMs.
Motu-client is used to circumvent the 1GB download limit.
The dates of the data to download are in dates.csv.

scripts can be executed in the terminal using:
!for file in files/*.sh; do sh "$file"; done
'''


import csv, fileinput, shutil, os, re

# open a csv file which has the list of dates and convert into list
with open('dates.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
data = data[:-2]

# create a motu-client script for each 13 month increment
text_to_search = "--date-min \"2018-11-16 00:00:00\" --date-max \"2018-12-16 12:00:00\""
count = 0
for i in range(0,len(data)-2, 13):
    start_date = data[i+13][0]
    end_date = data[i][0]
    replacement_text = "--date-min \""+ start_date +"\" --date-max \""+ end_date +"\""
    shutil.copyfile('script.txt', 'script(temp).txt')
    with fileinput.FileInput('script(temp).txt', inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace(text_to_search, replacement_text), end='')
    shutil.copyfile('script(temp).txt', './files/' + str(count) + '.sh')
    os.remove("script(temp).txt")
    count += 1

# change output .nc file names
fnames = os.listdir('./files')
text_to_search = 'testfile.nc'
for fname in fnames:
    pre, ext = os.path.splitext(fname)
    replacement_text = str(pre) + '.nc'
    with fileinput.FileInput('./files/'+fname, inplace=True) as file:
        for line in file:
            print(line.replace(text_to_search, replacement_text), end='')

# add #/bin/bash to beggining of each .sh file
def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

fnames = os.listdir('./files')
for fname in fnames:
    line_prepender('./files/'+fname, '#/bin/bash')
