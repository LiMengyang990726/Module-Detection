import pandas as pd

dis_bp = pd.read_table('/Users/limengyang/Workspaces/Module-Detection/FunctionalFeatures/dis_bp', delim_whitespace=True, names=('ProteinID', 'BP'))
dis_cc = pd.read_table('/Users/limengyang/Workspaces/Module-Detection/FunctionalFeatures/dis_cc', delim_whitespace=True, names=('ProteinID', 'CC'))
dis_mf = pd.read_table('/Users/limengyang/Workspaces/Module-Detection/FunctionalFeatures/dis_mf', delim_whitespace=True, names=('ProteinID', 'MF'))

end_bp = pd.read_table('/Users/limengyang/Workspaces/Module-Detection/FunctionalFeatures/end_bp', delim_whitespace=True, names=('ProteinID', 'BP'))
end_cc = pd.read_table('/Users/limengyang/Workspaces/Module-Detection/FunctionalFeatures/end_cc', delim_whitespace=True, names=('ProteinID', 'CC'))
end_mf = pd.read_table('/Users/limengyang/Workspaces/Module-Detection/FunctionalFeatures/end_mf', delim_whitespace=True, names=('ProteinID', 'MF'))

ndne_bp = pd.read_table('/Users/limengyang/Workspaces/Module-Detection/FunctionalFeatures/ndne_bp', delim_whitespace=True, names=('ProteinID', 'BP'))
ndne_cc = pd.read_table('/Users/limengyang/Workspaces/Module-Detection/FunctionalFeatures/ndne_cc', delim_whitespace=True, names=('ProteinID', 'CC'))
ndne_mf = pd.read_table('/Users/limengyang/Workspaces/Module-Detection/FunctionalFeatures/ndne_mf', delim_whitespace=True, names=('ProteinID', 'MF'))

dis_bp.shape
dis_cc.shape
dis_mf.shape

end_bp.shape
end_cc.shape
end_mf.shape

ndne_bp.shape
ndne_cc.shape
ndne_mf.shape

dis_bpcc = pd.merge(dis_bp,dis_cc,on='ProteinID',how = 'outer')
dis = pd.merge(dis_bpcc,dis_mf,on='ProteinID',how='outer')
dis.shape

end_bpcc = pd.merge(end_bp,end_cc,on='ProteinID',how = 'outer')
end = pd.merge(end_bpcc,end_mf,on='ProteinID',how='outer')
end.shape

ndne_bpcc = pd.merge(ndne_bp,ndne_cc,on='ProteinID',how = 'outer')
ndne = pd.merge(ndne_bpcc,ndne_mf,on='ProteinID',how='outer')
ndne.shape

frames = [dis,end,ndne]
result = pd.concat(frames)
result.shape

result_without_null = result.dropna()
result_without_null.shape

from sklearn import preprocessing
result_without_null['BP'] = preprocessing.scale(result_without_null['BP'])
result_without_null['CC'] = preprocessing.scale(result_without_null['CC'])
result_without_null['MF'] = preprocessing.scale(result_without_null['MF'])

result_without_null_norm = result_without_null
result_without_null_norm.to_csv("allFunctioinalFeatures.csv",index='ProteinID',sep=',')
