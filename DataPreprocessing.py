"""DataPreprocessing.py implements all kinds of preprocessing tools"""
import numpy as np
import pandas as pd


def data_loader(filename, featureNames=None, missing_value_symbol=None, headerRow=None, idCol=None):
    """Read the .data files for use in experiments."""
    df = pd.read_csv(filename, header=headerRow)
    if missing_value_symbol:
        f = lambda x: np.nan if x == missing_value_symbol else x
        df = df.applymap(f)
    if featureNames:
        df.columns = featureNames
    if idCol is not None:
        df.drop(idCol, inplace=True, axis=1)
    return df


def data_imputer(df, featureNames=None, featureKinds=None):
    """Given a dataset, imputes missing values with the feature (column) mean."""
    if featureNames is None:
        featureNames = list(df.columns)
    if featureKinds is None:
        featureKinds = ["d"] * len(featureNames)
    for i in range(len(featureNames)):
        featureName = featureNames[i]
        featureKind = featureKinds[i]
        col = df[featureName]
        if featureKind == "d":  # discret
            replaceValue = col.mode()[0]
        elif featureKind == "c":  # continuous
            replaceValue = col.mean()
        else:
            raise ValueError("Your value for featureKinds is not correct.")
        df[featureName].fillna(replaceValue, inplace=True)
    return df


def data_encode(df, featureName, type, ordinalList=None):
    """Given a dataset, encodes ordinal data as integers or performs one-hot encoding on nominal features."""
    cols = {}
    if type == "n":  # nominal
        assert ordinalList is None
        col = df[featureName]
        values = col.unique()
        for value in values:
            cols[str(value)] = pd.Series([0] * len(col), index=col.index)
        for i in col.index:
            value = col[i]
            cols[str(value)][i] = 1
        colNames = list(df.columns)
        NewColNames = []
        for c in cols:
            NewColName = "{}={}".format(str(featureName), str(c))
            df[NewColName] = cols[c]
            NewColNames.append(NewColName)
        df.drop(featureName, axis=1, inplace=True)
        index = colNames.index(featureName)
        df = df.reindex(columns=colNames[:index] + NewColNames + colNames[index + 1:])
    elif type == "o":  # ordinal
        assert ordinalList is not None
        col = df[featureName]
        f = lambda x: ordinalList.index(x)
        df[featureName] = col.map(f)
    else:
        raise ValueError("Your value for parameter 'type' is not correct")
    return df


def data_discretizer(df, featureName, type, width=None, frequency=None):
    """Given a dataset, discretizes real-valued features into discretized features."""
    if type == "ef":  # equal-frequency
        assert frequency is not None
        col = df[featureName]
        lst = list(col)
        lst.sort()
        f = lambda x: lst.index(x) // frequency
        df[featureName] = col.map(f).astype(int)
    elif type == "ew":  # equal-width
        assert width is not None
        col = df[featureName]
        f = lambda x: x // width
        df[featureName] = col.map(f).astype(int)
    else:
        raise ValueError("Your value for type is not correct")
    return df


def data_standardizer(df_train, featureName, df_test=None):
    """Given  a  training  and  test  set,  performs  z -score  standardization"""
    if isinstance(featureName,list):
        featureNames=featureName
        for feature in featureNames:
            if df_test is None:
                df_train=data_standardizer(df_train,feature)
            else:
                df_train,df_test=data_standardizer(df_train,feature,df_test=df_test)
        if df_test is None:
            return df_train
        else:
            return df_test,df_test
    else:
        col = df_train[featureName]
        col = col.astype(float) # some number in dataframe may be string!!!
        mean = col.mean()
        var = col.var()**0.5
        f = lambda x: (x - mean) / var
        df_train[featureName] = col.map(f)
        if df_test is None:
            return df_train
        else:
            df_test[featureName] = col.map(f)
            return df_train, df_test


def data_partitioner(df, ratioList, stratifyFeature=None, stratify_type="c"):
    """Given a dataset, first partitions the data randomly into 80% and 20% when the ratio list is [8,2]."""
    assert stratify_type == "c" or stratify_type == "r" # categorical or numerical
    if stratifyFeature is None:
        def convertList(ratioList,n):
            indexList=[0]
            x=0
            s=sum(ratioList)
            for i in ratioList[:-1]:
                x+=i
                indexList.append(n*x//s)
            indexList.append(n)
            return indexList
        returnDFs=[]
        n=df.shape[0]
        indexList=convertList(ratioList,n)
        df=df.sample(frac=1)
        for i in range(len(indexList)-1):
            returnDF=df.iloc[indexList[i]:indexList[i+1]-1]
            returnDFs.append(returnDF)
        return returnDFs
    elif stratify_type== "c":
        featureValues=df[stratifyFeature].unique()
        featureValues2=[]
        for featureValue in featureValues:
            dfForFeatureValue=df[df[stratifyFeature]==featureValue]
            featureValues2.append(data_partitioner(dfForFeatureValue,ratioList))
        returnDFS=[]
        for i in range(len(ratioList)):
            lst=[ x[i] for x in featureValues2 ]
            returnDF=pd.concat(lst)
            returnDF=returnDF.sample(frac=1)
            returnDFS.append(returnDF)
        return returnDFS
    elif stratify_type== "n":
        df=df.sort_values(by=stratifyFeature)
        returnDFS=[]
        n = df.shape[0]
        ratioList=[0]+ratioList
        s=sum(ratioList)
        for i in range(1,len(ratioList)):
            start=sum(ratioList[:i])
            end=sum(ratioList[:i+1])
            index_list=[i for i in range(n) if start <= i % s <= end]
            df_for_a_ratio=df.iloc[index_list]
            returnDFS.append(df_for_a_ratio)
        return returnDFS
    else:
        raise ValueError


def data_prediction_evaluator(groundTruth, predicted, metric=None, show_detail=False,show_more_detail=False):
    """given a set of ground truth and predicted values, calculates the appropriate, chosen evaluation
    metric (e.g., 0/1-loss or mean squared error)"""
    assert len(groundTruth)==len(predicted)
    assert metric=="01Loss" or metric=="MSE"
    if show_more_detail:
        print("The ground truth is:")
        print(groundTruth)
        print("The predicted is:")
        print(predicted)
    if metric == "01Loss":
        # groundTruth, predicted = groundTruth.align(predicted, axis=1, copy=False) # avoid a future warning
        boolLst = (groundTruth == predicted)
        boolLst = (boolLst == False)
        ans = sum(boolLst)
    elif metric == "MSE":
        n = len(predicted)
        diff = groundTruth - predicted
        diffSquared = diff*diff
        ans = sum(diffSquared) / n
    if show_detail:
        print("The {} is {}".format(metric, ans))
    return ans


def data_attribute_type_evaluator(df, threshold=30):
    """
    Evaluate the types of attributes in a DataFrame based on the number of unique values in each column.
    """
    attribute_kinds=[]
    for col in df.columns:
        if len(df[col].unique())<=threshold:
            attribute_kinds.append("c") # categorical
        else:
            attribute_kinds.append("n") # numerical
    return attribute_kinds

def data_label_rename(df,label_feature):
    labels = list(np.unique(df[label_feature]))
    df[label_feature]=df[label_feature].map(lambda x: labels.index(x))
    return df