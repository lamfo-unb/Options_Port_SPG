'''
                           Environment for the RL problem
'''
## ----- python SVM options
import pandas as pd
import numpy as np



#reading data
def readData(fileName,columnNames,indexes,assetNames):
    data = pd.read_csv(fileName)
    
    #selecting assets
    data = data[data['Security_x'].str.startswith(assetNames)]
    data.dropna(subset = indexes, inplace = True)
    #creating double indexing for data and security
    data.set_index(indexes, inplace=True)

    data = data[columnNames]
    data.sort_index(inplace=True)
    return data

filename = 'RL_optionsBasePrototipoCall.csv'
states = ['Value', 'STRIKE_PRC', 'CLOSE_x','CLOSE_F1']
indexNames = ['Security_x','dia_F1']
assets = ('AAPLA191810000.U','AAPLA191810500.U','AAPLA191811000.U','AAPLA191811500.U','AAPLA191812000.U')
priceColumn = 'CLOSE_F1'
dataset = readData(filename,states,indexNames,assets)    



class Environment():
    
    def __init__(self,df):
        self.df = df
        self.days = 0
    
    
    #creates return column
    def createReturn(self,priceColumn,assets):
        for ind in assets:
            self.df.loc[ind,'Return'] = self.df.loc[ind,priceColumn].pct_change(1).values
        self.df.dropna(inplace=True)



    def removeSingles(self,k):
        temp = self.df.swaplevel()
        temp.sort_index(inplace = True)
        count = temp.groupby(temp.index.names[0]).size()
        row_names = count[count > k]
        self.df2 = temp.loc[list(row_names.index)]
        return self.df2


    def getReward(self,w,datas):
        R = np.dot(self.df2.loc[datas[self.days],'Return'],w)
        self.days+=1
        return R
    
env = Environment(dataset)
env.createReturn(priceColumn,assets)
dados = env.removeSingles(1)


def sliceData(df, startDate, endDate):
    #slice data according to start and end dates
    temp = df.loc[startDate:endDate]
    # getting days as a list 
    lst = temp.index.values.tolist()
    lst2 = [item[0] for item in lst]
    dates = list(dict.fromkeys(lst2))
    return temp,dates

data,dates = sliceData(dados,'2016-10-24','2016-10-28')
