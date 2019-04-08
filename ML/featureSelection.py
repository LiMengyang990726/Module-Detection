import ML.dataPreparation as dataPreparation
import statsmodels.formula.api as sm
import numpy as np

################# Read Input
path = "/Users/limengyang/Workspaces/Module-Detection/"
data = dataPreparation.readInput(path)
target = data['Target'][1:].values


def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)

    regressor_OLS.summary()
    return x, columns


SL = 0.05
data_modeled, selected_columns = backwardElimination(data.iloc[:, 1:37].values, data.iloc[:, 37].values, SL,
                                                     target)