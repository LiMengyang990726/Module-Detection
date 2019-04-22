import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

################# Read Input
train_1 = pd.read_csv("DS1_train.csv")
train_1.drop(['Unnamed: 0'],axis = 1, inplace = True)
test_1 = pd.read_csv("DS1_test.csv")
test_1.drop(["Unnamed: 0"],axis = 1, inplace = True)
train_2 = pd.read_csv("DS2_train.csv")
train_2.drop(["Unnamed: 0"],axis = 1, inplace = True)
test_2 = pd.read_csv("DS2_test.csv")
test_2.drop(["Unnamed: 0"],axis = 1, inplace = True)
train_3 = pd.read_csv("DS3_train.csv")
train_3.drop(["Unnamed: 0"],axis = 1, inplace = True)
test_3 = pd.read_csv("DS3_test.csv")
test_3.drop(["Unnamed: 0"],axis = 1, inplace = True)

frame_1 = [train_1, test_1]
dataset_1 = pd.concat(frame_1)
frame_2 = [train_2, test_2]
dataset_2 = pd.concat(frame_2)
frame_3 = [train_3, test_3]
dataset_3 = pd.concat(frame_3)



###################################################
#
# Method 1: Using Independent T test
#
###################################################

nondisease_1 = dataset_1[(dataset_1['Target'] == 0)]
nondisease_1.reset_index(inplace=True)

disease_1 = dataset_1[(dataset_1['Target'] == 1)]
disease_1.reset_index(inplace=True)

f = open("dataset_1_feature_indepedent_t_test.txt","w+")

for i in range(2, 38):
    #
    diff = nondisease_1.iloc[:,i] - disease_1.iloc[:,i]
    #
    title = nondisease_1.columns[i] + " correlation"
    xlabel = nondisease_1.columns[i] + " value"
    figure_name = nondisease_1.columns[i] + " Correlation.png"
    diff.plot(kind="hist", title=title)
    plt.xlabel(xlabel)
    plt.savefig(figure_name)
    #
    result = stats.ttest_ind(nondisease_1.iloc[:,i],
                             disease_1.iloc[:,i])
    f.write(str(result)+"\n")

f.close()



nondisease_2 = dataset_2[(dataset_2['Target'] == 0)]
nondisease_2.reset_index(inplace=True)

disease_2 = dataset_2[(dataset_2['Target'] == 1)]
disease_2.reset_index(inplace=True)

f = open("dataset_2_feature_indepedent_t_test.txt","w+")

for i in range(2, 37):
    #
    diff = nondisease_2.iloc[:,i] - disease_2.iloc[:,i]
    #
    title = nondisease_2.columns[i] + " correlation"
    xlabel = nondisease_2.columns[i] + " value"
    figure_name = nondisease_2.columns[i] + " Correlation.png"
    diff.plot(kind="hist", title=title)
    plt.xlabel(xlabel)
    plt.savefig(figure_name)
    #
    result = stats.ttest_ind(nondisease_2.iloc[:,i],
                             disease_2.iloc[:,i])
    f.write(str(result)+"\n")

f.close()



nondisease_3 = dataset_3[(dataset_3['Target'] == 0)]
nondisease_3.reset_index(inplace=True)

disease_3 = dataset_3[(dataset_3['Target'] == 1)]
disease_3.reset_index(inplace=True)

f = open("dataset_3_feature_indepedent_t_test.txt","w+")

for i in range(2, 37):
    #
    diff = nondisease_3.iloc[:,i] - disease_3.iloc[:,i]
    #
    title = nondisease_3.columns[i] + " correlation"
    xlabel = nondisease_3.columns[i] + " value"
    figure_name = nondisease_3.columns[i] + " Correlation.png"
    diff.plot(kind="hist", title=title)
    plt.xlabel(xlabel)
    plt.savefig(figure_name)
    #
    result = stats.ttest_ind(nondisease_3.iloc[:,i],
                             disease_3.iloc[:,i])
    f.write(str(result)+"\n")

f.close()



###################################################
#
# Method 2: Using Mann Whitney U Test
#
###################################################

nondisease_1 = dataset_1[(dataset_1['Target'] == 0)]
nondisease_1.reset_index(inplace=True)

disease_1 = dataset_1[(dataset_1['Target'] == 1)]
disease_1.reset_index(inplace=True)

f = open("dataset_1_feature_mann_whitney_u_test.txt","w+")

for i in range(2, 38):
    result = mannwhitneyu(nondisease_1.iloc[:,i], disease_1.iloc[:,i])
    f.write(str(result)+"\n")

f.close()



nondisease_2 = dataset_2[(dataset_2['Target'] == 0)]
nondisease_2.reset_index(inplace=True)

disease_2 = dataset_2[(dataset_2['Target'] == 1)]
disease_2.reset_index(inplace=True)

f = open("dataset_2_feature_mann_whitney_u_test.txt","w+")

for i in range(2, 38):
    result = mannwhitneyu(nondisease_2.iloc[:,i], disease_2.iloc[:,i])
    f.write(str(result)+"\n")

f.close()


nondisease_3 = dataset_3[(dataset_3['Target'] == 0)]
nondisease_3.reset_index(inplace=True)

disease_3 = dataset_3[(dataset_3['Target'] == 1)]
disease_3.reset_index(inplace=True)

f = open("dataset_3_feature_mann_whitney_u_test.txt","w+")

for i in range(2, 38):
    result = mannwhitneyu(nondisease_3.iloc[:,i], disease_3.iloc[:,i])
    f.write(str(result)+"\n")

f.close()

###################################################
#
# Method 3: Feature Importance
#
###################################################

################# Prepare individual features
target_train_1 = train_1['Target'].tolist()
target_train_2 = train_2['Target'].tolist()
target_train_3 = train_3['Target'].tolist()

target_test_1 = test_1['Target'].tolist()
target_test_2 = test_2['Target'].tolist()
target_test_3 = test_3['Target'].tolist()

################# Get the feature importance on the best model

# {'min_samples_split': 5, 'max_features': 'auto', 'max_depth': 93,
#   'min_samples_leaf': 1, 'bootstrap': False, 'n_estimators': 1000}
# {'min_samples_split': 5, 'max_features': 'auto', 'max_depth': 16,
#   'min_samples_leaf': 1, 'bootstrap': False, 'n_estimators': 900}
# {'min_samples_split': 5, 'max_features': 'sqrt', 'max_depth': 93,
#   'min_samples_leaf': 4, 'bootstrap': False, 'n_estimators': 500}

model_1 = RandomForestClassifier(n_estimators=1000, bootstrap=False, min_samples_leaf=1, max_depth=93,
                                 max_features="auto", min_samples_split=5)

model_2 = RandomForestClassifier(n_estimators=900, bootstrap=False, min_samples_leaf=1, max_depth=16,
                                 max_features="auto", min_samples_split=5)

model_3 = RandomForestClassifier(n_estimators=500, bootstrap=False, min_samples_leaf=4, max_depth=93,
                                 max_features="sqrt", min_samples_split=5)

model_1.fit(train_1.drop(['Target','ProteinID'],axis=1), target_train_1)
model_2.fit(train_2.drop(['Target','ProteinID'],axis=1), target_train_2)
model_3.fit(train_3.drop(['Target','ProteinID'],axis=1), target_train_3)

model_1.feature_importances_
model_2.feature_importances_
model_3.feature_importances_


