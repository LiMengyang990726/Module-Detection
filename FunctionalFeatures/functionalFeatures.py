import pandas as pd
import os

# Input path
inputPath = "/Users/limengyang/Workspaces/Module-Detection/data/dataset2/"
dis_bp = pd.read_table(os.path.join(inputPath,'dis_bp'), delim_whitespace=True, names=('ProteinID', 'BP'))
dis_cc = pd.read_table(os.path.join(inputPath,'dis_cc'), delim_whitespace=True, names=('ProteinID', 'CC'))
dis_mf = pd.read_table(os.path.join(inputPath,'dis_mf'), delim_whitespace=True, names=('ProteinID', 'MF'))

end_bp = pd.read_table(os.path.join(inputPath,'end_bp'), delim_whitespace=True, names=('ProteinID', 'BP'))
end_cc = pd.read_table(os.path.join(inputPath,'end_cc'), delim_whitespace=True, names=('ProteinID', 'CC'))
end_mf = pd.read_table(os.path.join(inputPath,'end_mf'), delim_whitespace=True, names=('ProteinID', 'MF'))

ndne_bp = pd.read_table(os.path.join(inputPath,'ndne_bp'), delim_whitespace=True, names=('ProteinID', 'BP'))
ndne_cc = pd.read_table(os.path.join(inputPath,'ndne_cc'), delim_whitespace=True, names=('ProteinID', 'CC'))
ndne_mf = pd.read_table(os.path.join(inputPath,'ndne_mf'), delim_whitespace=True, names=('ProteinID', 'MF'))


dis_bpcc = pd.merge(dis_bp,dis_cc,on='ProteinID',how = 'outer')
dis = pd.merge(dis_bpcc,dis_mf,on='ProteinID',how='outer')
dis.head()
end_bpcc = pd.merge(end_bp,end_cc,on='ProteinID',how = 'outer')
end = pd.merge(end_bpcc,end_mf,on='ProteinID',how='outer')
end.head()
ndne_bpcc = pd.merge(ndne_bp,ndne_cc,on='ProteinID',how = 'outer')
ndne = pd.merge(ndne_bpcc,ndne_mf,on='ProteinID',how='outer')
ndne.head()

frames = [dis,end,ndne]
allFunctioinalFeatures = pd.concat(frames)
allFunctioinalFeatures.dropna()
allFunctioinalFeatures.to_csv("allFunctionalFeatures.csv",index='ProteinID',sep=',')
