import pandas as pd
import numpy as np
import pyreadr
import datetime
import multiprocessing as mp
import psutil
from PLSR_modeling import random_CV,spatial_CV,leave_one_out_CV,site_extropolation,leave_one_climate_zone_out,cross_PFT_CV,\
leave_one_PFT_out,cross_sites_PFTs,PFT_extropolation

climate = pd.read_csv('all_WorldClim_variables.csv')
climate.drop(['latitude','longitude'],axis = 1,inplace = True)

all_refl = pd.read_csv('all_reflectance.csv')
all_traits = pd.read_csv('all_traits.csv', dtype = {'Sample date': object})

index = ['datasets','Site_num','Sample date','usda_code','English Name','Latin Name','PFT','latitude',
         'longitude','Chla+b','Ccar','EWT','LMA','Cant']
all_traits = all_traits[index]
all_traits = pd.concat([all_traits,climate],axis = 1)

climate_variables= pd.DataFrame(np.zeros(shape = (len(all_traits['Site_num'].unique()),3)),columns = ['Site_num','MAT','MAP'])
k = 0
for i in all_traits['Site_num'].unique():
    df = all_traits[all_traits['Site_num'] == i]
    df1 = df.loc[:,'bio01':]
    mat = df1.mean()['bio01']
    ma_p = df1.mean()['bio12']
    climate_variables.iloc[k] = np.array([i,mat,ma_p])
    k = k+1
climate_variables['MAT'] = climate_variables['MAT'].astype(float)
climate_variables['MAP'] = climate_variables['MAP'].astype(float)
climate_variables['MAP'] = climate_variables['MAP']/10

result = pyreadr.read_r("Whittaker_biomes.rda")
ddf = result['Whittaker_biomes']
ddf['biome'].unique()
ddf.reset_index(drop = True, inplace  = True)
ddf.columns = ["Mean Annual Temperature (°C)",'Mean Annual Precipitation (cm)','biome_id','(a) Whittaker biomes']

inx = [4,3,4,4,4,4,4,4,4,4,3,3,3,3,4,4,3,4,7,7,3,4,4,4,7,1,5,4,4,3,3,4,3,3,4,4,7,7,7,7,6,7,7,4,4,4,4,4,4,4,4,4,4,
       4,1,1,3,0,1,3,3,4,4,7,4,3,4,4,3,7,8,4,4,4,4,3,3,2,3,4,4,4,4,4,4,4,5,5,5,5,7,7,6]
c_zones = []
for i in inx:
    temp = ddf['(a) Whittaker biomes'].unique()[i]
    c_zones.append(temp)
climate_variables['zones'] = c_zones

def data_for_use(tr,all_traits,all_refl):
    traits = all_traits[all_traits[tr]>0]
    traits = traits[(traits['datasets'] != 'Dataset#3')&(traits['datasets'] != 'Dataset#4')&
                    (traits['datasets'] != 'Dataset#5')&(traits['datasets'] != 'Dataset#6')&
                    (traits['datasets'] != 'Dataset#12')&(traits['datasets'] != 'Dataset#13')]
    for i in traits['Site_num'].unique():
        temp = list(climate_variables[climate_variables['Site_num']==i]['zones'])[0]
        traits.loc[traits['Site_num']==i,'zones'] = temp
    print(tr,len(traits),'samples')

    refl = all_refl.iloc[traits.index]
    refl = refl.loc[:,'450':'2400']
    refl.reset_index(drop = True,inplace = True)
    traits.reset_index(drop = True,inplace = True)

    if (tr =='EWT')|(tr =='LMA'):
        traits[tr] = traits[tr]*10000
        
    X = refl
    y = traits
    return X,y

start_t = datetime.datetime.now()
print('start:', start_t)
print('cores',psutil.cpu_count(logical=False))
pool = mp.Pool(psutil.cpu_count(logical=False))
trait_name = ['Chla+b','Ccar']#,'EWT','LMA']

functions = [random_CV,spatial_CV,leave_one_out_CV,site_extropolation,leave_one_climate_zone_out,cross_PFT_CV,
             leave_one_PFT_out,cross_sites_PFTs,PFT_extropolation]

for tr in trait_name:
    for func in functions:
        X,y = data_for_use(tr,all_traits,all_refl)
        if (func == random_CV)|(func == spatial_CV):
            args=(X,y,tr,10)
        elif func == cross_PFT_CV:
            args=(X,y,tr,10)
        else:
            args=(X,y,tr)
        
        p = pool.apply_async(func=func,args= args)
pool.close()
pool.join()

end_t = datetime.datetime.now()
elapsed_sec = (end_t - start_t).total_seconds()
print('end:', end_t)
print('total:',elapsed_sec/60, 'min')