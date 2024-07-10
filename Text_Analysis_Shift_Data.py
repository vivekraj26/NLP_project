#!/usr/bin/env python
# coding: utf-8

# In[332]:


import numpy as np
import pandas as pd 
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
class textAnalysis:
    def __init__(self, df):
        self.df = df 
         
    def get_required_data(self):
        df0 = self.df.copy()
        df0.dropna(inplace = True)
        df0.drop_duplicates(inplace = True)
        df0.drop(columns='id',inplace=True)
        df0.drop(columns='Parameter Name',inplace=True)
        df0.drop(columns='Lemma_Description',inplace=True)
        return df0

    def unique_statement_freq(self):
        #Method to find unique statements
        return self.df.Description.value_counts().to_frame()

    def get_preprocess_data(self):
        #self.get_required_data()
        df0 = self.get_required_data()
        corpus = []
        lemma = WordNetLemmatizer()
        for i in range(len(df0)):
            unt = re.sub('[^0-9a-zA-Z]',' ',df0.iloc[i,3])
            #unt.append(unit_1.iloc[i,3].lower())
            unt = unt.lower()
            unt = unt.split()
            unt = [lemma.lemmatize(word) for word in unt if word not in set(stopwords.words('english'))]
            unt = ' '.join(unt)
            corpus.append(unt)
        df0["clean_description"] =  corpus
        return df0

    def get_unit(self,unit):
        tc = self.get_preprocess_data()
        units_grp=tc.groupby("Unit",group_keys=True)
        ut = units_grp.get_group(unit)
        return ut
    def get_shiftwise_words(self,unit):
        df1 = self.get_unit(unit)
        shifts = df1.groupby("Shift",group_keys=True)
        ngt_sfts = shifts.get_group("Night Shift")
        mor_sfts = shifts.get_group("Morning Shift")
        eve_sfts = shifts.get_group("Evening Shift")
        nt_sft = []
        night  = []
        for i in ngt_sfts.clean_description:
            nt_sft.append(i.split())
        for i in nt_sft:
            for j in i:
                night.append(j)
        mr_sft = []
        morning  = []
        for i in mor_sfts.clean_description:
            mr_sft.append(i.split())
        for i in mr_sft:
            for j in i:
                morning.append(j)
        ev_sft = []
        evening  = []
        for i in eve_sfts.clean_description:
            ev_sft.append(i.split())
        for i in ev_sft:
            for j in i:
                evening.append(j)
        sfts = pd.DataFrame()
        sfts["eveningwords"] = evening
        sfts["nightwords"] = night
        sfts["morningwords"] = morning
        return sfts
    
    def get_shiftwise_word_freq(self,unit):
        #method to find shift wise word frequency
        w = self.get_shiftwise_words(unit)
        nightsft_wrdcount = w.nightwords.value_counts().to_frame()
        morningsft_wrdcount =w.morningwords.value_counts().to_frame()
        eveningsft_wrdcount =w.eveningwords.value_counts().to_frame()
        return nightsft_wrdcount,morningsft_wrdcount,eveningsft_wrdcount

    def unitwise_equipments(self,unit):
        df2 = self.get_unit(unit)
        unitx = []
        unitxwords  = []
        for i in df2.clean_description:
            unitx.append(i.split())
        for i in unitx:
            for j in i:
                unitxwords.append(j)
        ptern = r'\d+[a-z]+\d+[a-z]+\b|\d+[a-z]+\d+\b'
        instru = []
        equip = []
        for i in unitxwords:
            match = re.findall(ptern, i)
            if len(match) != 0:
                #print(match)
                instru.append(match)
        equip = pd.DataFrame()
        equip["equipments"] = instru
        return equip
    
    def catch_and_append_equipments_in_df(self,unit):
        df3 = self.get_unit(unit)
        ptern = r'\d+[a-z]+\d+[a-z]+\b|\d+[a-z]+\d+\b'
        df3["equipments"] = df3["clean_description"].apply(lambda x: ''.join(re.findall(ptern, x)))
        return df3
    
    def get_equipmentwise_problem(self,unit,equipment):
        df4 =self.catch_and_append_equipments_in_df(unit)
        grouped_by_equipment=df4.groupby("equipments",group_keys=True)
        equipment_wise_problem = grouped_by_equipment.get_group(equipment)
        return equipment_wise_problem
    
    def get_equipment_frequency(self,unit):
        #Method to find most frequently mentioned equipment
        df4 =self.catch_and_append_equipments_in_df(unit)
        return df4.equipments.value_counts().to_frame()
    
    def plot_word_cloud(self,unit):
        #Method to plot word cloud of equipments
        df4 =self.catch_and_append_equipments_in_df(unit)
        Equipment_combined = ' '.join(df4['equipments'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(Equipment_combined)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def sequential_call(self,unit,equipment):
        self.get_required_data()
        self.unique_statement_freq()
        self.get_preprocess_data()
        self.get_unit(unit)
        self.get_shiftwise_words(unit)
        self.get_shiftwise_word_freq(unit)
        self.unitwise_equipments(unit)
        self.catch_and_append_equipments_in_df(unit)
        self.get_equipmentwise_problem(unit,equipment)
        self.get_equipment_frequency(unit)
        self.plot_word_cloud(unit)


# In[333]:


df = pd.read_csv('data.csv')
df


# In[334]:


obj1 = textAnalysis(df)
obj1.get_required_data()


# In[335]:


obj1.unique_statement_freq()


# In[336]:


obj1.get_preprocess_data()


# In[337]:


obj1.get_unit("CDU-I")


# In[338]:


obj1.get_shiftwise_words("CDU-I")


# In[339]:


obj1.get_shiftwise_word_freq('CDU-I')


# In[340]:


obj1.unitwise_equipments("CDU-I")


# In[341]:


obj1.catch_and_append_equipments_in_df('CDU-I')


# In[342]:


obj1.get_equipmentwise_problem("CDU-I","2f2")


# In[343]:


obj1.get_equipment_frequency("CDU-I")


# In[344]:


obj1.plot_word_cloud("CDU-I")


# In[345]:


obj1.sequential_call("CDU-I","2f2")


# In[ ]:





# In[ ]:




