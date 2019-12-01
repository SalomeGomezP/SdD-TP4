# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:36:55 2019

@author: Moi
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import math
from sklearn.metrics import mean_squared_error
# =============================================================================
# import données
# =============================================================================
def get_data(nom, test) :
    suffixe=""
    data=[]
    if (test==True) :
        suffixe=".test"
    else :
        suffixe=".base"
    
    with open ("MovieLens/100K/ml-100k/"+nom+suffixe, "r") as fileHandler:
        for line in fileHandler:
            row = line.split("	")
            data.append(row)
    retour=pd.DataFrame(data, columns=["userid", "itemid", "rating", "timestamp"])
    retour['rating'] = retour['rating'].astype('int32')
    retour['userid'] = retour['userid'].astype('int32')
    retour['itemid'] = retour['itemid'].astype('int32')
    return retour


data_train_1=get_data("u1",False).drop(columns="timestamp")
data_train_2=get_data("u2",False).drop(columns="timestamp")
data_train_3=get_data("u3",False).drop(columns="timestamp")
data_train_4=get_data("u4",False).drop(columns="timestamp")
data_train_5=get_data("u5",False).drop(columns="timestamp")
data_test_1=get_data("u1",True).drop(columns="timestamp")
data_test_2=get_data("u2",True).drop(columns="timestamp")
data_test_3=get_data("u3",True).drop(columns="timestamp")
data_test_4=get_data("u4",True).drop(columns="timestamp")
data_test_5=get_data("u5",True).drop(columns="timestamp")

data_test_1=data_test_1.dropna()
data_test_2=data_test_2.dropna()
data_test_3=data_test_3.dropna()
data_test_4=data_test_4.dropna()
data_test_5=data_test_5.dropna()


# =============================================================================
# normalisation des données
# =============================================================================

def normaliser(data) :
    val=list(data.userid.unique())
    for i in range (1,len(val)+1) :
        data_user=data.loc[data['userid'] == i]
        moy=data_user.mean(axis = 0).get_value('rating', 0)
        data_user['rating']=data_user['rating']-moy
        data.loc[data['userid'] == i]=data_user
    return data


#data_train_1=normaliser(data_train_1)
#data_train_2=normaliser(data_train_2)
#data_train_3=normaliser(data_train_3)
#data_train_4=normaliser(data_train_4)
#data_train_5=normaliser(data_train_5)
data_test_1=normaliser(data_test_1)
data_test_2=normaliser(data_test_2)
data_test_3=normaliser(data_test_3)
data_test_4=normaliser(data_test_4)
data_test_5=normaliser(data_test_5)


def convList(utilisateur, user_id):
    liste = np.empty(1682)
    liste[:] = np.nan
    liste=list(liste)
    retour=pd.DataFrame(data=liste, columns=[user_id] )
    items=utilisateur.itemid.unique()
    for item in items :
        tmp=utilisateur.loc[utilisateur['itemid'] == item]
        tmp=tmp.iloc[0]['rating']
        retour.at[item-1,user_id]=tmp
    return retour
    

#conv=convList(data_train_1.loc[data_train_1['userid'] == 1],1)


def conversionEnsemble(data) :
    data=data_train_1
    retour=pd.DataFrame()
    val=list(data.userid.unique())
    for i in range (1,len(val)+1) :
        print(i)
        data_user=data.loc[data['userid'] == i]
        tmp=convList(data_user,i)
        if (i==1):
            retour=tmp
        else:
            retour=retour.join(tmp)
    
    return retour

#data_train_1_conv=conversionEnsemble(data_train_1)
#data_train_2_conv=conversionEnsemble(data_train_2)
#data_train_3_conv=conversionEnsemble(data_train_3)
#data_train_4_conv=conversionEnsemble(data_train_4)
#data_train_5_conv=conversionEnsemble(data_train_5)
#data_train_1_conv.to_csv('data_train_1_conv.csv')
#data_train_2_conv.to_csv('data_train_2_conv.csv')
#data_train_3_conv.to_csv('data_train_3_conv.csv')
#data_train_4_conv.to_csv('data_train_4_conv.csv')
#data_train_5_conv.to_csv('data_train_5_conv.csv')

data_train_1_conv=pd.read_csv('data_train_1_conv.csv')
data_train_2_conv=pd.read_csv('data_train_2_conv.csv')
data_train_3_conv=pd.read_csv('data_train_3_conv.csv')
data_train_4_conv=pd.read_csv('data_train_4_conv.csv')
data_train_5_conv=pd.read_csv('data_train_5_conv.csv')

def score(utilisateur, item, data_utilisateur_conv) :
    
    #calcul similarités entre utilisateur et data_utilisateur
    utilisateur_conv=convList(utilisateur, "utilisateur")
    df=utilisateur_conv.join(data_utilisateur_conv)
    similarites=df.corr(method='pearson')
    similarites=similarites['utilisateur']
    similarites=similarites[1:]
    
    #calcul denominateur
    valsabsolues=similarites.abs()
    denominateur=valsabsolues.sum()
    
    #calcul nominateur
    sim=similarites.to_numpy()
    scores=df.drop(columns='utilisateur').loc[item-1,:].to_numpy()
    sim=np.nan_to_num(sim)
    scores=np.nan_to_num(scores)
    
    score=sim.dot(scores.T)
    score=score/denominateur
    return score

#socre=score(data_test_1.loc[data_test_1['userid'] == 1], 1, data_train_1_conv)


def evaluation_test(data_train_conv,data_test):
    
    rating_reels=[]
    rating_predits=[]
    
    users=data_test.userid.unique()
    for user in users :
        donnees1=data_test.loc[data_test['userid'] == user]
        items=donnees1.itemid.unique()
        for item in items :        
            print(item)
            a=score(donnees1, item,data_train_conv)
            
            tmp=donnees1.loc[donnees1['itemid'] == item]
            tmp=tmp.iloc[0]['rating']
            rating_reels.append(tmp)
            rating_predits.append(a)
        
        #calcul erreur
        erreur=mean_squared_error(rating_reels,rating_predits)
        return erreur

erreur=evaluation_test(data_train_5_conv, data_test_5)
        
