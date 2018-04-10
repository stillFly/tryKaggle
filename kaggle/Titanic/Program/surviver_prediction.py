# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 11:23:43 2018

@title: Titanic surviver prediction

@author: Steal Flaining
"""
#PassengerId£ºÒ»¸öÓÃÒÔ±ê¼ÇÃ¿¸ö³Ë¿ÍµÄÊý×Öid
#Survived£º±ê¼Ç³Ë¿ÍÊÇ·ñÐÒ´æ¡ª¡ªÐÒ´æ(1)¡¢ËÀÍö(0)¡£ÎÒÃÇ½«Ô¤²âÕâÒ»ÁÐ
#Pclass£º±ê¼Ç³Ë¿ÍËùÊô´¬²ã¡ª¡ªµÚÒ»²ã(1),µÚ¶þ²ã(2),µÚÈý²ã(3)
#Name£º³Ë¿ÍÃû×Ö
#Sex£º³Ë¿ÍÐÔ±ð¡ª¡ªÄÐmale¡¢Å®female
#Age£º³Ë¿ÍÄêÁä²¿·Ö
#SibSp£º´¬ÉÏÐÖµÜ½ãÃÃºÍÅäÅ¼µÄÊýÁ¿
#Parch£º´¬ÉÏ¸¸Ä¸ºÍº¢×ÓµÄÊýÁ¿
#Ticket£º³Ë¿ÍµÄ´¬Æ±ºÅÂë
#Fare£º³Ë¿ÍÎª´¬Æ±¸¶ÁË¶àÉÙÇ®
#Cabin£º³Ë¿Í×¡ÔÚÄÄ¸ö´¬²Õ
#Embarked£º³Ë¿Í´ÓÄÄ¸öµØ·½µÇÉÏÌ©Ì¹Äá¿ËºÅ

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

if __name__ == "__main__":
    csvFile = pd.read_csv("E:/kaggle/Titanic/Data/train.csv")
    #print(dataAnalysis(csvFile))
    # fill missing value of Age
    csvFile['Age'] = csvFile['Age'].fillna(csvFile['Age'].median())
    # transport column Sex: male == 0; female == 1
    csvFile.loc[csvFile['Sex']=='male', 'Sex'] = 0
    csvFile.loc[csvFile['Sex']=='female', 'Sex'] = 1
    # fill missing value of Embarked
    csvFile['Embarked'] = csvFile['Embarked'].fillna('S')
    # transport column Embarked: S == 0; C == 1; Q == 2
    csvFile.loc[csvFile['Embarked']=='S', 'Embarked'] = 0
    csvFile.loc[csvFile['Embarked']=='C', 'Embarked'] = 1
    csvFile.loc[csvFile['Embarked']=='Q', 'Embarked'] = 2
    # training classifier
    predictor = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    kf = KFold(n_splits = 3, random_state = 1)
    for data in kf.split(csvFile['Sex']):
        print(data)
