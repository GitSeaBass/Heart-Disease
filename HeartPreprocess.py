import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, RFE, SequentialFeatureSelector
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

def main():
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restECG', 'thalach', 'ExAng', 'OldPeak', 'Slope', 'Ca', 'Thal', 'Num']
    full_column_names = []
    for i in range(76):
        full_column_names.append(str(i + 1))

    #creating dataframes from datasets
    cleve_filename = './heart+disease/processed.cleveland.data'
    cleve_data = pd.read_csv(cleve_filename, header=None, names=column_names)
    replace_nan(cleve_data)
    cleve_data = pd.DataFrame(remove_missing(cleve_data), columns=column_names)
    cleve_X = cleve_data.iloc[:,:-1]
    cleve_Y = cleve_data.iloc[:,-1]

    hun_filename = './heart+disease/processed.hungarian.data'
    hun_data = pd.read_csv(hun_filename, header=None)
    replace_nan(hun_data)
    hun_data = pd.DataFrame(remove_missing(hun_data), columns=column_names)
    hun_X = hun_data.iloc[:,:-1]
    hun_Y = hun_data.iloc[:,-1]

    switz_filename = './heart+disease/processed.switzerland.data'
    switz_data = pd.read_csv(switz_filename, header=None)
    replace_nan(switz_data)
    switz_data = pd.DataFrame(remove_missing(switz_data), columns=column_names)
    switz_X = switz_data.iloc[:,:-1]
    switz_Y = switz_data.iloc[:,-1]

    va_filename = './heart+disease/processed.va.data'
    va_data = pd.read_csv(va_filename, header=None)
    replace_nan(va_data)
    va_data = pd.DataFrame(remove_missing(va_data), columns=column_names)
    va_X = va_data.iloc[:,:-1]
    va_Y = va_data.iloc[:,-1]

    frames = [cleve_data, hun_data, switz_data, va_data]
    all = pd.concat(frames)
    all_X = all.iloc[:,:-1]
    all_Y = all.iloc[:,-1]

    #unprocessed switzerland
    full_switz = './heart+disease/fix-swit.txt'
    full = pd.read_csv(full_switz, header=None, delim_whitespace=True)
    replace_nan(full)
    full = pd.DataFrame(remove_missing(full), columns=full_column_names)

    full_y = full.iloc[:,57]
    full_x = full.drop(columns=['58','75', '76'])
    full_column_names.remove('58')
    full_column_names.remove('75')
    full_column_names.remove('76')

    #recursive_feature_elimination_full(full_x, full_y, full_column_names)

    #print number of instances in each dataset
    #print(f"c_len: {cleve_data.shape}, h_len: {hun_data.shape}, s_len: {switz_data.shape}, v_len: {va_data.shape}")
    
    #VT on dataset, currently not removing any features
    #print(variance_feature_selection(cleve_data))
    #print(variance_feature_selection(full_x))

    #RFE on different datasets
    
    #getting the ranks of all variables
    #ranks = []
    #for _ in range(50):
    #    ranks.append(recursive_feature_elimination(all_X,all_Y))

    #means = np.array(ranks).mean(axis=0)
    #print(ranks)
    #print(means)

    trimmed_all = all.drop(columns = ['fbs', 'Slope', 'chol', 'trestbps'])
    #pd.concat([trimmed_all, all_Y], axis = 1)
    print(trimmed_all)
    trimmed_all.to_csv('./heart+disease/feature-selected-processed.all.data', index=False)

    #recursive_feature_elimination(cleve_X, cleve_Y)
    #recursive_feature_elimination(hun_X, hun_Y)
    #recursive_feature_elimination(switz_X, switz_Y)
    #recursive_feature_elimination(va_X, va_Y)
    #recursive_feature_elimination(all_X, all_Y)

    #PCA on datasets
    #principal_component(hun_X, hun_Y, num_components=2)

    #forward/backward elimination
    #forward_elim(cleve_X, cleve_Y, dir='forward')
    #forward_elim(cleve_X, cleve_Y, dir='backward')
    #forward_elim(all_X, all_Y, dir='forward')
    #forward_elim(all_X, all_Y, dir='backward')

#replaces the '?' with np.nan to allow for imputation
def replace_nan(df):
    df.replace('?', np.nan, inplace=True)

#Imputer
def remove_missing(df):
    imp = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
    imp.fit(df)
    return imp.transform(df)
    
def variance_feature_selection(df):
    scaler = StandardScaler()
    df = scaler.fit_transform(df)

    sel = VarianceThreshold()
    df = sel.fit_transform(df)
    print(df.shape)
    return df

def recursive_feature_elimination(df_X, df_Y):
    scaler = StandardScaler()

    #df_X[['age', 'trestbps', 'chol', 'thalach']] = scaler.fit_transform(df_X[['age', 'trestbps', 'chol', 'thalach']])
    df_X[['age', 'trestbps', 'chol', 'thalach', 'sex', 'cp', 'fbs', 'restECG', 'OldPeak', 'Slope', 'Thal', 'Ca', 'ExAng']] = scaler.fit_transform(df_X[['age', 'trestbps', 'chol', 'thalach', 'sex', 'cp', 'fbs', 'restECG', 'OldPeak', 'Slope', 'Thal', 'Ca', 'ExAng']])
    print(df_X)

    Xscaled = pd.DataFrame(df_X, columns = df_X.columns)
    df_Y = df_Y.astype('int')

    #train test split
    x_train, x_test, y_train, y_test = train_test_split(Xscaled, df_Y, test_size=.2)

    logreg = linear_model.LogisticRegression()

    n = int(len(df_X.columns)/2)
    
    selector = RFE(logreg, n_features_to_select=1, step=1)
    selector = selector.fit(x_train, y_train)
    
    print('')
    print(f'Features in X: {len(df_X.columns)}, {df_X.columns}')
    print(f'Size of selector.ranking_: {len(selector.ranking_)}, {selector.ranking_}')
    print(f'Size of selector.support_: {len(selector.support_)}, {selector.support_}')
    print(f'Features in fitted estimator (excluding bias): {len(selector.estimator_.coef_)}, {selector.estimator_.coef_}')

    df = pd.DataFrame({'Column':df_X.columns, 'Included':selector.support_, 'Rank':selector.ranking_})
    coefs = list(df[df['Included'] == True].Column)
    est = selector.estimator_
    s = f'{selector.estimator_.intercept_[0]:.3f} '
    for c in range(len(coefs)):
        s = s + f'+ {selector.estimator_.coef_[0][c]:.3f}[{coefs[c]}]'
    
    print(s)
    
    print(f'Size of selector.support_: {len(selector.support_)}, {selector.support_}')
    print(accuracy_score(y_test, selector.predict(x_test)))


    rankings = np.array([int(x) for x in selector.ranking_])

    return rankings

def recursive_feature_elimination_full(df_X, df_Y, col):
    scaler = StandardScaler()

    #df_X[['age', 'trestbps', 'chol', 'thalach']] = scaler.fit_transform(df_X[['age', 'trestbps', 'chol', 'thalach']])
    #df_X[['age', 'trestbps', 'chol', 'thalach', 'sex', 'cp', 'fbs', 'restECG', 'OldPeak', 'Slope', 'Thal', 'Ca', 'ExAng']] = scaler.fit_transform(df_X[['age', 'trestbps', 'chol', 'thalach', 'sex', 'cp', 'fbs', 'restECG', 'OldPeak', 'Slope', 'Thal', 'Ca', 'ExAng']])
    print(df_X)
    
    df_X = scaler.fit_transform(df_X)

    df_X = pd.DataFrame(df_X, columns=col)
    Xscaled = pd.DataFrame(df_X, columns = col)
    df_Y = df_Y.astype('int')

    #train test split
    x_train, x_test, y_train, y_test = train_test_split(Xscaled, df_Y, test_size=.2)

    logreg = linear_model.LogisticRegression()

    n = int(len(df_X.columns)/2)
    
    selector = RFE(logreg, n_features_to_select=14, step=1)
    selector = selector.fit(x_train, y_train)
    
    print('')
    print(f'Features in X: {len(df_X.columns)}, {df_X.columns}')
    print(f'Size of selector.ranking_: {len(selector.ranking_)}, {selector.ranking_}')
    print(f'Size of selector.support_: {len(selector.support_)}, {selector.support_}')
    print(f'Features in fitted estimator (excluding bias): {len(selector.estimator_.coef_)}, {selector.estimator_.coef_}')

    df = pd.DataFrame({'Column':df_X.columns, 'Included':selector.support_, 'Rank':selector.ranking_})
    coefs = list(df[df['Included'] == True].Column)
    est = selector.estimator_
    s = f'{selector.estimator_.intercept_[0]:.3f} '
    for c in range(len(coefs)):
        s = s + f'+ {selector.estimator_.coef_[0][c]:.3f}[{coefs[c]}]'
    
    print(s)
    
    print(f'Size of selector.support_: {len(selector.support_)}, {selector.support_}')
    print(accuracy_score(y_test, selector.predict(x_test)))

def principal_component(df_X, df_Y, num_components = 2):
    sc = MinMaxScaler()
    scaled_X = sc.fit_transform(df_X)
    
    pca = PCA(n_components = num_components)
    components = pca.fit_transform(scaled_X)
    component_df = pd.DataFrame(components)
    component_df['target'] = df_Y

    xpca = component_df[component_df.columns[0:-1]]
    ypca = component_df[component_df.columns[-1]]
    dtree = DecisionTreeClassifier()
    scores = cross_val_score(dtree, xpca, ypca, cv=StratifiedKFold(shuffle=True))
    print(f'Mean acc (of CV scores), transformed data:  {np.mean(scores):.3f}')
    
def forward_elim(df_X, df_Y, dir):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_X[['age', 'trestbps', 'chol', 'thalach', 'sex', 'cp', 'fbs', 'restECG', 'OldPeak', 'Slope', 'Thal', 'Ca', 'ExAng']])

    #train test split
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, df_Y, test_size=.2)

    #knn = KNeighborsClassifier(n_neighbors=10)
    #sfs = SequentialFeatureSelector(knn, n_features_to_select=7, direction=dir)
    #sfs.fit(x_train, y_train)

    #print(sfs.get_support())
    #print(sfs.transform(x_train).shape)

    logreg = linear_model.LogisticRegression()
    logreg.fit(x_train, y_train)

    print(accuracy_score(y_test, logreg.predict(x_test)))

def basic(df_X, df_Y):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_X[['age', 'trestbps', 'chol', 'thalach', 'sex', 'cp', 'fbs', 'restECG', 'OldPeak', 'Slope', 'Thal', 'Ca', 'ExAng']])

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, df_Y, test_size=.2)

    logreg = linear_model.LogisticRegression()
    logreg.fit(x_train, y_train)

    return (accuracy_score(y_test, logreg.predict(x_test)))
main()