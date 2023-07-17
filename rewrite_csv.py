from pathlib import Path
import numpy as np
import os
from random import randint
import pandas as pd


images = os.listdir('Normal_Skin/')
dx_data = ['confocal', 'consensus', 'follow_up', 'histo']
image_id = [Path(img).stem for img in images]
age = [randint(20, 70) for i in range(len(image_id))]
sex_int = [randint(0,1) for i in range(len(image_id))]
sex = ['female' if s==0 else 'male' for s in sex_int]
Dx_type_int = [randint(0,3) for i in range(len(image_id))]
Dx_type = [dx_data[i] for i in Dx_type_int]

Dx = pd.Series(['ns'] * len(image_id))
localization = pd.Series(['hand'] * len(image_id))


data = {'lesion_id': image_id, 'image_id':image_id, 'dx': Dx, 'dx_type':Dx_type,  'age':age, 'sex':sex,'localization':localization}
df_hand = pd.DataFrame(data)

df_skin = pd.read_csv('HAM10000_metadata.csv')
df_hand = df_hand.reset_index(drop=True)
df_skin = pd.concat([df_skin, df_hand], ignore_index=True)

# Write back to CSV 
df_skin.to_csv('HAM10000_metadata_new.csv', index=False)