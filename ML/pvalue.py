import ML.dataPreparation as dataPreparation
from scipy.stats import mannwhitneyu
import operator

################# Read Input
path = "/Users/limengyang/Workspaces/Module-Detection/"
data = dataPreparation.readInput(path)

################# Store result
result = {}

################# Get the column to correlate to
target = data['Target']

################# Get All Features and p-value
avgSP = data['Average Shortest Path to all Disease genes']
temp = mannwhitneyu(avgSP, target)
result['avgSP'] = temp[1]

betweennessCentrality = data['BetweennessCentrality']
temp = mannwhitneyu(betweennessCentrality, target)
result['betweennessCentrality'] = temp[1]

closenessCentrality = data['ClosenessCentrality']
temp = mannwhitneyu(closenessCentrality, target)
result['closenessCentrality'] = temp[1]

degreeCentrality = data['DegreeCentrality']
temp = mannwhitneyu(degreeCentrality, target)
result['degreeCentrality'] = temp[1]

eigenvectorCentrality = data['EigenvectorCentrality']
temp = mannwhitneyu(eigenvectorCentrality, target)
result['eigenvectorCentrality'] = temp[1]

harmonicCentrality = data['HarmonicCentrality']
temp = mannwhitneyu(harmonicCentrality, target)
result['harmonicCentrality'] = temp[1]

modularity = data['Modularity']
temp = mannwhitneyu(modularity, target)
result['modularity'] = temp[1]

pageRank = data['PageRank']
temp = mannwhitneyu(pageRank, target)
result['pageRank'] = temp[1]


bp = data['BP']
temp = mannwhitneyu(bp, target)
result['bp'] = temp[1]

cc = data['CC']
temp = mannwhitneyu(cc, target)
result['cc'] = temp[1]

mf = data['MF']
temp = mannwhitneyu(mf, target)
result['mf'] = temp[1]


frequencyA = data['FrequencyA']
temp = mannwhitneyu(frequencyA, target)
result['frequencyA'] = temp[1]

frequencyC = data['FrequencyC']
temp = mannwhitneyu(frequencyC, target)
result['frequencyC'] = temp[1]

frequencyD = data['FrequencyD']
temp = mannwhitneyu(frequencyD, target)
result['frequencyD'] = temp[1]

frequencyE = data['FrequencyE']
temp = mannwhitneyu(frequencyE, target)
result['frequencyE'] = temp[1]

frequencyF = data['FrequencyF']
temp = mannwhitneyu(frequencyF, target)
result['frequencyF'] = temp[1]

frequencyG = data['FrequencyG']
temp = mannwhitneyu(frequencyG, target)
result['frequencyG'] = temp[1]

frequencyH = data['FrequencyH']
temp = mannwhitneyu(frequencyH, target)
result['frequencyH'] = temp[1]

frequencyI = data['FrequencyI']
temp = mannwhitneyu(frequencyI, target)
result['frequencyI'] = temp[1]

frequencyK = data['FrequencyK']
temp = mannwhitneyu(frequencyK, target)
result['frequencyK'] = temp[1]

frequencyL = data['FrequencyL']
temp = mannwhitneyu(frequencyL, target)
result['frequencyL'] = temp[1]

frequencyM = data['FrequencyM']
temp = mannwhitneyu(frequencyM, target)
result['frequencyM'] = temp[1]

frequencyN = data['FrequencyN']
temp = mannwhitneyu(frequencyN, target)
result['frequencyN'] = temp[1]

frequencyP = data['FrequencyP']
temp = mannwhitneyu(frequencyP, target)
result['frequencyP'] = temp[1]

frequencyQ = data['FrequencyQ']
temp = mannwhitneyu(frequencyQ, target)
result['frequencyQ'] = temp[1]

frequencyR = data['FrequencyR']
temp = mannwhitneyu(frequencyR, target)
result['frequencyR'] = temp[1]

frequencyS = data['FrequencyS']
temp = mannwhitneyu(frequencyS, target)
result['frequencyS'] = temp[1]

frequencyT = data['FrequencyT']
temp = mannwhitneyu(frequencyT, target)
result['frequencyT'] = temp[1]

frequencyV = data['FrequencyV']
temp = mannwhitneyu(frequencyV, target)
result['frequencyV'] = temp[1]

frequencyW = data['FrequencyW']
temp = mannwhitneyu(frequencyW, target)
result['frequencyW'] = temp[1]

frequencyY = data['FrequencyY']
temp = mannwhitneyu(frequencyY, target)
result['frequencyY'] = temp[1]


aromaticity = data['Aromaticity']
temp = mannwhitneyu(aromaticity, target)
result['aromaticity'] = temp[1]

isoelectric = data['Isoelectric']
temp = mannwhitneyu(isoelectric, target)
result['isoelectric'] = temp[1]

ssfractionHelix = data['SSfractionHelix']
temp = mannwhitneyu(ssfractionHelix, target)
result['ssfractionHelix'] = temp[1]

ssfractionTurn = data['SSfractionTurn']
temp = mannwhitneyu(ssfractionTurn, target)
result['ssfractionTurn'] = temp[1]

ssfractionSheet = data['SSfractionSheet']
temp = mannwhitneyu(ssfractionSheet, target)
result['ssfractionSheet'] = temp[1]


################# Get all correlation
sorted_result = sorted(result.items(), key=operator.itemgetter(1))
print(sorted_result)
