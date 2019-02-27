import pandas as pd
import os
from sklearn import preprocessing
pd.options.display.max_columns = None

def dataCleaning(path):

    # Get Input
    absolutePath = path
    ffeatures = pd.read_csv(os.path.join(absolutePath,"allFunctionalFeatures.csv"))
    sfeatures = pd.read_csv(os.path.join(absolutePath,"allSequenceFeatures.csv"))
    tfeatures = pd.read_csv(os.path.join(absolutePath,"allTopoFeatures.csv"))

    # Basic cleaning
    ffeatures.drop(['Unnamed: 0'], axis = 1, inplace = True)
    ffeatures.dropna(inplace = True)
    ffeatures.shape

    sfeatures.drop(['Unnamed: 0','FrequencyB','FrequencyJ','FrequencyO','FrequencyU','FrequencyX','FrequencyZ','Hydropathy','Instability'],axis = 1, inplace = True)
    sfeatures.dropna(inplace = True)
    sfeatures.shape

    # As there are many NaN in Local clustering coefficient, I temporarily dropped this column to ensure sufficient amount of data
    tfeatures.drop(['Unnamed: 0','Local Clustering Coefficient'],axis = 1, inplace = True)
    tfeatures.dropna(inplace = True)
    tfeatures.shape

    result = pd.merge(ffeatures,sfeatures,on='ProteinID',how='left')
    result = pd.merge(tfeatures,result,on='ProteinID',how='left')
    result.dropna(inplace = True)
    result.shape

    result.head()

    # Process SSfraction tuple
    tup_all = result['SSfraction']
    Helix = []
    Turn = []
    Sheet = []

    tup_all = tup_all.tolist()

    for i in range(len(tup_all)):
        tup = list(tup_all[i])
        tup = tup[1:len(tup)-1]

        counter = 0
        start = [0] * 3
        end = [0] *3

        helix_each = []
        turn_each = []
        sheet_each = []

        for i in range(len(tup)):
            if(tup[i] == ','):
                start[counter+1] = i
                end[counter] = (i-1)
                counter += 1

        helix_each = tup[0:end[0]+1]
        helix_float = "".join(helix_each)
        helix_float = float(helix_float)
        Helix.append(helix_float)

        turn_each = tup[start[1]+2:end[1]+1]
        turn_float = "".join(turn_each)
        turn_float = float(turn_float)
        Turn.append(turn_float)

        sheet_each = tup[start[2]+2:]
        sheet_float = "".join(turn_each)
        sheet_float = float(sheet_float)
        Sheet.append(sheet_float)

    result['SSfractionHelix'] = Helix
    result['SSfractionTurn'] = Turn
    result['SSfractionSheet'] = Sheet
    result.drop('SSfraction',axis=1,inplace=True)

    # Normalization
    result['FrequencyA'] = preprocessing.scale(result['FrequencyA'])
    result['FrequencyC'] = preprocessing.scale(result['FrequencyC'])
    result['FrequencyD'] = preprocessing.scale(result['FrequencyD'])
    result['FrequencyE'] = preprocessing.scale(result['FrequencyE'])
    result['FrequencyF'] = preprocessing.scale(result['FrequencyF'])
    result['FrequencyG'] = preprocessing.scale(result['FrequencyG'])
    result['FrequencyH'] = preprocessing.scale(result['FrequencyH'])
    result['FrequencyI'] = preprocessing.scale(result['FrequencyI'])
    result['FrequencyK'] = preprocessing.scale(result['FrequencyK'])
    result['FrequencyL'] = preprocessing.scale(result['FrequencyL'])
    result['FrequencyM'] = preprocessing.scale(result['FrequencyM'])
    result['FrequencyN'] = preprocessing.scale(result['FrequencyN'])
    result['FrequencyP'] = preprocessing.scale(result['FrequencyP'])
    result['FrequencyQ'] = preprocessing.scale(result['FrequencyQ'])
    result['FrequencyR'] = preprocessing.scale(result['FrequencyR'])
    result['FrequencyS'] = preprocessing.scale(result['FrequencyS'])
    result['FrequencyT'] = preprocessing.scale(result['FrequencyT'])
    result['FrequencyV'] = preprocessing.scale(result['FrequencyV'])
    result['FrequencyW'] = preprocessing.scale(result['FrequencyW'])
    result['FrequencyY'] = preprocessing.scale(result['FrequencyY'])
    result['Aromaticity'] = preprocessing.scale(result['Aromaticity'])
    result['Isoelectric'] = preprocessing.scale(result['Isoelectric'])
    result['Average Shortest Path to all Disease genes'] = preprocessing.scale(result['Average Shortest Path to all Disease genes'])
    result['BetweennessCentrality'] = preprocessing.scale(result['BetweennessCentrality'])
    result['ClosenessCentrality'] = preprocessing.scale(result['ClosenessCentrality'])
    result['DegreeCentrality'] = preprocessing.scale(result['DegreeCentrality'])
    result['EigenvectorCentrality'] = preprocessing.scale(result['EigenvectorCentrality'])
    result['HarmonicCentrality'] = preprocessing.scale(result['HarmonicCentrality'])
    # result['Local Clustering Coefficient'] = preprocessing.scale(result['Local Clustering Coefficient'])
    result['Modularity'] = preprocessing.scale(result['Modularity'])
    result['PageRank'] = preprocessing.scale(result['PageRank'])
    result['SSfractionHelix'] = preprocessing.scale(result['SSfractionHelix'])
    result['SSfractionTurn'] = preprocessing.scale(result['SSfractionTurn'])
    result['SSfractionSheet'] = preprocessing.scale(result['SSfractionSheet'])
    result['BP'] = preprocessing.scale(result['BP'])
    result['CC'] = preprocessing.scale(result['CC'])
    result['MF'] = preprocessing.scale(result['MF'])

    # Stage 1 Output
    result.to_csv(os.path.join(absolutePath,"cleanFeatures.csv"),index='ProteinID',sep=',')

    # Add Target column
    data = pd.read_csv(os.path.join(absolutePath,"cleanFeatures.csv"))
    data.drop(['Unnamed: 0'],axis = 1, inplace = True)
    disease = pd.read_csv(os.path.join(absolutePath,"data/dataset2/disgenes_uniprot.csv"),names = ['ProteinID'])
    disease = disease['ProteinID'].tolist()

    data['Target'] = ""

    for index, row in data.iterrows():
        if(row['ProteinID'] in disease):
            # print(row['ProteinID'] + "in")
            data.at[index, 'Target'] = 1
        else:
            # print("not in")
            data.at[index, 'Target'] = 0

    data.groupby('Target').count()
    data.head()

    # Final Output
    data.to_csv(os.path.join(absolutePath,"cleanFeatures.csv"),index = 'ProteinID', sep = ',')

if __name__ == "__main__":

    path = "/Users/limengyang/Workspaces/Module-Detection/"
    dataCleaning(path)
