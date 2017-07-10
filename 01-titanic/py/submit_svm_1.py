# This script shows you how to make a submission using a few
# useful Python libraries.
# It gets a public leaderboard score of 0.76077.
# Maybe you can tweak it and do better...?

import pandas as pd
# import xgboost as xgb
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np

# Load the data
train_df = pd.read_csv('./data/train.csv', header=0)
test_df = pd.read_csv('./data/test.csv', header=0)
total_df = pd.concat((train_df,test_df),axis=0,ignore_index=True)

# データの項目追加

# Class : 敬称判定
def name1_classifier(name_df):
    name_class_df = pd.DataFrame(columns={'honorific'})
    
    for name in name_df:
        if 'Miss' in name:
            df = pd.DataFrame({'honorific':['miss']})
        elif 'Mrs' in name:
            df = pd.DataFrame({'honorific':['mrs']})
        elif 'Master' in name:
            df = pd.DataFrame({'honorific':['master']})
        elif 'Mr' in name:
            df = pd.DataFrame({'honorific':['mr']})
        else :
            df = pd.DataFrame({'honorific':[np.nan]})
        name_class_df = name_class_df.append(df,ignore_index=True)
    return name_class_df

# Class : 敬称判定(二値表示形式)
def name2_classifier(name_df):
    name_class_df = pd.DataFrame(columns={'miss','mrs','master','mr'})
    
    for name in name_df:
        if 'Miss' in name:
            df = pd.DataFrame({'miss':[1],'mrs':[0],'master':[0],'mr':[0]})
        elif 'Mrs' in name:
            df = pd.DataFrame({'miss':[0],'mrs':[1],'master':[0],'mr':[0]})
        elif 'Master' in name:
            df = pd.DataFrame({'miss':[0],'mrs':[0],'master':[1],'mr':[0]})
        elif 'Mr' in name:
            df = pd.DataFrame({'miss':[0],'mrs':[0],'master':[0],'mr':[1]})
        else :
            df = pd.DataFrame({'miss':[0],'mrs':[0],'master':[0],'mr':[0]})
        name_class_df = name_class_df.append(df,ignore_index=True)
    return name_class_df

# Class : 年齢推定(敬称毎の中央値代入)
def age_classmedian(input_df):
    age_class_df = pd.DataFrame(columns={'Age2'})
    
    for i, v in input_df.iterrows():
        if np.isnan(v['Age']) :
            if v['honorific'] == 'mr':
                df = pd.DataFrame({'Age2':[30.0]})
            elif v['honorific'] == 'miss':
                df = pd.DataFrame({'Age2':[21.0]})
            elif v['honorific'] == 'master':
                df = pd.DataFrame({'Age2':[3.5]})
            elif v['honorific'] == 'mrs':
                df = pd.DataFrame({'Age2':[35.0]})
            else :
                df = pd.DataFrame({'Age2':[28.0]})
        else :
            df = pd.DataFrame({'Age2':[v['Age']]})
        age_class_df = age_class_df.append(df,ignore_index=True)
    return age_class_df

# Class : 単身者，家族連れの判定
def family_classifier(input_df):
    family_class_df = pd.DataFrame(columns={'alone','family','family(child)'})
    
    for i, v in input_df.iterrows():
        if v['SibSp'] == 0 and v['Parch'] == 0 :
            # 単身者
            df = pd.DataFrame({'alone':[1],'family':[0],'family(child)':[0]})
        elif v['SibSp'] >= 1 and v['Parch'] == 0 :
            # 家族連れ（子供なし）
            df = pd.DataFrame({'alone':[0],'family':[1],'family(child)':[0]})
        elif v['SibSp'] >= 0 and v['Parch'] >= 1 :
            # 家族連れ（子供あり）
            df = pd.DataFrame({'alone':[0],'family':[0],'family(child)':[1]})
        else :
            # その他
            df = pd.DataFrame({'alone':[0],'family':[0],'family(child)':[0]})
        family_class_df = family_class_df.append(df,ignore_index=True)
    return family_class_df

# We'll impute missing values using the median for numeric columns and the most
# common value for string columns.
# This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

# 名前から敬称を抽出
name_dum = name1_classifier(total_df['Name'])
total_df_name_dum = pd.concat((total_df,name_dum),axis=1)

# 名前から敬称を抽出(二値表示形式)
name_dum = name2_classifier(total_df_name_dum['Name'])
total_df_name_dum = pd.concat((total_df_name_dum,name_dum),axis=1)

# Sex:性別を二値表示
# sex_dum = pd.get_dummies(total_df_name_dum['Sex'])
# total_df_name_dum = pd.concat((total_df_name_dum,sex_dum),axis=1)

# Age:欠損値の補完
age_dum = age_classmedian(total_df_name_dum[['Age','honorific']])
total_df_name_dum = pd.concat((total_df_name_dum,age_dum['Age2']),axis=1)

# 単身者，家族連れの判定
family_dum = family_classifier(total_df_name_dum[['SibSp','Parch']])
total_df_name_dum = pd.concat((total_df_name_dum,family_dum[['alone','family','family(child)']]),axis=1)

# Fareの欠損を中央値で補完
total_df_name_dum = DataFrameImputer().fit_transform(total_df_name_dum)

# 機械学習の対象とするカラム
feature_columns_to_use = ['Pclass','Sex','Age2','miss','mrs','master','mr','alone','family','family(child)']
nonnumeric_columns = ['Sex']
big_X_imputed = total_df_name_dum[feature_columns_to_use]

# XGBoost doesn't (yet) handle categorical features automatically, so we need to change
# them to columns of integer values.
# See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more
# details and options
le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

# Prepare the inputs for the model
train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
train_y = train_df['Survived']

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
#gbm = xgb.XGBClassifier(max_depth=8, n_estimators=5000, learning_rate=0.05).fit(train_X, train_y)
#predictions = gbm.predict(test_X)

i = 0.01
j = 2.48
clf = svm.SVC(kernel='rbf',random_state=0,gamma=i,C=j).fit(train_X, train_y)
predictions = clf.predict(test_X)

# Kaggle needs the submission to have a certain format;
# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
# for an example of what it's supposed to look like.

submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)

