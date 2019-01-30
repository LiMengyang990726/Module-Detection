import timeit
start= timeit.default_timer()
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParamData import kd
from scipy.stats.stats import pearsonr


##########################################################################################
#
# Sequence Features Functions
#
##########################################################################################
# Return a dictionary
def getAAfreq(seq):  #returns a dictionary
    analyse = ProteinAnalysis(seq)
    x=analyse.get_amino_acids_percent()
    return x

# Return single number
def getAromaticity(seq):
    analyse = ProteinAnalysis(seq)
    x=analyse.aromaticity()
    return x

# Return single number
def getIsoelectric(seq):
    analyse = ProteinAnalysis(seq)
    x=analyse.isoelectric_point()
    return x

# Return single number
def getHydropathy(seq):
    analyse = ProteinAnalysis(seq)
    x=analyse.gravy()
    return x

# Return single number
def getInstability(seq):
    analyse = ProteinAnalysis(seq)
    x=analyse.instability_index()
    return x

# Return three values
def getSSfraction(seq): #returns a tuple of three floats (Helix, Turn, Sheet)
    analyse = ProteinAnalysis(seq)
    x=analyse.secondary_structure_fraction()
    return x

def getAAscale(seq):
    analyse = ProteinAnalysis(seq)
    x=analyse.protein_scale(param_dict=kd, window=9, edge=1.0)
    return x

def getPCC(aa1,aa2): #returns pcc and p-value, input is dictionary

    a1_val = []
    a2_val = []

    for key in aa1:
        a1_val.append(aa1[key])
        a2_val.append(aa2[key])

    pc = pearsonr(a1_val,a2_val)
    return pc

def writeFile(fh,sent):
    fh.write(sent)

def getDis(v1,v2):
    dis=v1-v2
    adis=abs(dis)
    return adis


##########################################################################################
#
# Main Code
#
#
##########################################################################################
#INPUT FILES
seq_file="./SequenceFeatures/endgenes_sequence.txt"

#read ids and sequences in a dictionary
seq_data=np.genfromtxt(seq_file, dtype=str, delimiter='\t')

import csv
f = open('SequenceFeaturesEnd.csv', mode='w')
fieldnames = ['Protein_ID', 'FrequencyA','FrequencyB',
                'FrequencyC','FrequencyD','FrequencyE',
                'FrequencyF','FrequencyG','FrequencyH',
                'FrequencyI','FrequencyJ','FrequencyK',
                'FrequencyL','FrequencyM','FrequencyN',
                'FrequencyO','FrequencyP','FrequencyQ',
                'FrequencyR','FrequencyS','FrequencyT',
                'FrequencyU','FrequencyV','FrequencyW',
                'FrequencyX','FrequencyY','FrequencyZ',
                'Aromaticity','Isoelectric','Hydropathy',
                'Instability','SSfraction']
f_writer = csv.DictWriter(f, fieldnames=fieldnames)

for p in range(len(seq_data)):
    protein_id = seq_data[p][0]
    seq = seq_data[p][1]

    frequency = getAAfreq(seq)
    frequencies = [0] * 26
    for key,value in frequency.items():
        alpha = ord(key) - 65
        frequencies[alpha] = value

    aromaticity = getAromaticity(seq)
    isoelectric = getIsoelectric(seq)
    # hydropathy = getHydropathy(seq)
    # instability = getInstability(seq)
    ssfraction = getSSfraction(seq)
    f_writer.writerow({'Protein_ID': protein_id,
                        'FrequencyA': frequencies[0],
                        'FrequencyB': frequencies[1],
                        'FrequencyC': frequencies[2],
                        'FrequencyD': frequencies[3],
                        'FrequencyE': frequencies[4],
                        'FrequencyF': frequencies[5],
                        'FrequencyG': frequencies[6],
                        'FrequencyH': frequencies[7],
                        'FrequencyI': frequencies[8],
                        'FrequencyJ': frequencies[9],
                        'FrequencyK': frequencies[10],
                        'FrequencyL': frequencies[11],
                        'FrequencyM': frequencies[12],
                        'FrequencyN': frequencies[13],
                        'FrequencyO': frequencies[14],
                        'FrequencyP': frequencies[15],
                        'FrequencyQ': frequencies[16],
                        'FrequencyR': frequencies[17],
                        'FrequencyS': frequencies[18],
                        'FrequencyT': frequencies[19],
                        'FrequencyU': frequencies[20],
                        'FrequencyV': frequencies[21],
                        'FrequencyW': frequencies[22],
                        'FrequencyX': frequencies[23],
                        'FrequencyY': frequencies[24],
                        'FrequencyZ': frequencies[25],
                        'Aromaticity': aromaticity,
                        'Isoelectric': isoelectric,
                        # 'Hydropathy': hydropathy,
                        # 'Instability': instability,
                        'SSfraction': ssfraction})
f.close()

# stop= timeit.default_timer()
# print("runtime: "+str(stop-start))
