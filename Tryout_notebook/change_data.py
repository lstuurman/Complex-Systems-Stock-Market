import glob
import os
import re

# files = glob.glob('../twitter_data/*/*/*.pkl')
# print(files)

# # delete all pickle files : 
# for f in files:
#     os.remove(f)

# now convert text files to field/date.txt
# f_names = glob.glob('../twitter_data/*/*/*.txt')
# for f in f_names:
#     #print(f)
#     #date = re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2})", f)
#     print('new_name :' + f[:-25] + '.txt')
#     os.rename(f,f[:-25] + '.txt')