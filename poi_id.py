#!/usr/bin/python

sys.path.append("../final_project/")
import sys
import pickle
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest,mutual_info_classif

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#explore the data
print('There are {} people in the dataset.'.format(len(data_dict)))
print('There are {} features.'.format(len(list(data_dict.values())[0])))

#list the features
print(list(list(data_dict.values())[0].keys()))

# Features with missing values
nan_counts_poi = defaultdict(int)
nan_counts_non_poi = defaultdict(int)
for data_point in data_dict.values():
    if data_point['poi'] == True:
        for feature, value in data_point.items():
            if value == "NaN":
                nan_counts_poi[feature] += 1
    elif data_point['poi'] == False:
        for feature, value in data_point.items():
            if value == "NaN":
                nan_counts_non_poi[feature] += 1
    else:
        print('Got an uncategorized person.')
nan_counts_df = pd.DataFrame([nan_counts_poi, nan_counts_non_poi]).T
nan_counts_df = nan_counts_df.fillna(value=0)
nan_counts_df.columns = ['# NaN in POIs', '# NaN in non-POIs']
nan_counts_df['# NaN total'] = nan_counts_df['# NaN in POIs'] + \
    nan_counts_df['# NaN in non-POIs']
nan_counts_df['% NaN in POIs'] = nan_counts_df['# NaN in POIs'] / \
    poi_counts[True] * 100
nan_counts_df['% NaN in non-POIs'] = nan_counts_df['# NaN in non-POIs'] / \
    poi_counts[False] * 100
nan_counts_df['% NaN total'] = nan_counts_df['# NaN total'] / \
    len(data_dict) * 100

print nan_counts_df

features_list = [i for i in list(list(data_dict.values())[0].keys()) if i not in ['poi', 'name', 'email_address']]  #its the feature ist used

### Task 2: Remove outliers
"""
    There're two kinds of outliers,one is a data point with name 'TOTAL', which is the sum of other points and I removed this 'TOTAL' data point from my analysis.
    The other is of other features. Since these values are accurate and some of them are acutally POIs, I retained these data points.

"""
def allFeaturesFormat(data_dict):
    temp_list = []
    for name, features_values in data_dict.items():
        temp_dict = {}
        temp_dict['name'] = name
        for feature, value in features_values.items():
            if feature in ['poi', 'email_address']:
                temp_dict[feature] = value
            else:
                if value == 'NaN':
                    value = 0
                temp_dict[feature] = float(value)
        temp_list.append(temp_dict)
    df = pd.DataFrame(temp_list)
    return df


df = allFeaturesFormat(data_dict)
g = sns.FacetGrid(df, hue='poi', size=7)
g.map(plt.scatter, 'salary', 'bonus', alpha=.7)
g.add_legend()
plt.show()

data_dict.pop('TOTAL', 0)


### Task 3: Create new feature(s)
"""
I added 2 features that might help increase the predictability:
    1. pcnt_from_poi = from_poi_to_this_person / to_messages
    2. pcnt_to_poi = from_this_person_to_poi / to_messages
    
"""

def computeFraction(poi_messages, all_messages):
    fraction = 0
    if poi_messages != "NaN" and all_messages != "NaN":
        fraction = poi_messages / all_messages
    return fraction

for data_point in data_dict.values():
    # Create feature pcnt_from_poi
    from_poi_to_this_person = data_point['from_poi_to_this_person']
    to_messages = data_point['to_messages']
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
    data_point['pcnt_from_poi'] = fraction_from_poi
    
    # Create feature pcnt_from_poi
    from_this_person_to_poi = data_point['from_this_person_to_poi']
    from_messages = data_point['from_messages']
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
    data_point['pcnt_to_poi'] = fraction_to_poi


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing

df = allFeaturesFormat(my_dataset)
labels, features = df.poi,df.drop(['poi', 'name', 'email_address'], axis=1)
features_list = [i for i in list(features.columns)]

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
ft2 = scaler.fit_transform(features) # do minmac scalar to all numerical features to make sure they are in the same scale


### select best features


def get_k_best(labels, features,features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
        """
    k_best = SelectKBest(mutual_info_classif,k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    return k_best_features


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

###4.1  Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression

l_clf = LogisticRegression(tol = 0.0001, C = 10**-8, penalty = 'l2', random_state = 42)

###4.2 Support Vector Machine Classifier
from sklearn.svm import SVC
s_clf = SVC(kernel='rbf', C=1000,gamma = 0.0001,random_state = 42, class_weight = 'auto')

###4.3 Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(max_depth = 6,max_features = 'sqrt',n_estimators = 10, random_state = 42)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation


def evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3):
    print clf
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        if trial % 10 == 0:
            if first:
                sys.stdout.write('\nProcessing')
            sys.stdout.write('.')
            sys.stdout.flush()
            first = False

    print "done.\n"
    print "precision: {}".format(mean(precision))
    print "recall:    {}".format(mean(recall))
    return mean(precision), mean(recall)

##logistic
for c in [3000,2000,1000]:
    for k in [8,10,12,14,16,18,20]:
        print "C = " + str(c) + " and k = " + str(k)
        fl = get_k_best(labels, ft2,features_list, k)
        bf = [i for i in fl.keys() if fl[i] > 0]
        ft3 = scaler.fit_transform(features[bf])
        l_clf = LogisticRegression(tol = 0.001, C = c, penalty = 'l2', random_state = 42,verbose = 0)
        evaluate_clf(l_clf, ft3, labels, num_iters=100, test_size=0.3)
#C = 2000 and k = 8 is best

##SVM
for c in [100,1,0.1]:
    for g in [10,1,0.1,0.01,0.001]:
        for k in [10,12,14,16]:
            print "C = " + str(c) + " G = " + str(g) +" and k = " + str(k)
            fl = get_k_best(labels, ft2,features_list, k)
            bf = [i for i in fl.keys() if fl[i] > 0]
            ft3 = scaler.fit_transform(features[bf])
            s_clf = SVC(kernel='rbf', C=c,gamma = g,random_state = 42, class_weight = 'balanced')
            evaluate_clf(s_clf, ft3, labels, num_iters=100, test_size=0.3)
#C = 100 G = 0.1 and k = 16 is best

##random forest
for c in [2,4,6,8,10,12,14]:
    for n in [10,15,20,30,40,50]:
        for k in [5,8,10,12,14,16]:
            print "C = " + str(c) + " N = " + str(n) +" and k = " + str(k)
            fl = get_k_best(labels, ft2,features_list, k)
            bf = [i for i in fl.keys() if fl[i] > 0]
            ft3 = scaler.fit_transform(features[bf])
            rf_clf = RandomForestClassifier(max_depth = c,max_features = 'sqrt',n_estimators = n, random_state = 42)
            evaluate_clf(rf_clf, features, labels, num_iters=10, test_size=0.3)
#C = 10 N = 15 and k = 16 is best


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
s_clf = SVC(kernel='rbf', C=100,gamma = 0.1,random_state = 42, class_weight = 'balanced')
fl = get_k_best(labels, ft2,features_list, 16)
bf = [i for i in fl.keys() if fl[i] > 0]
ft3 = scaler.fit_transform(features[bf])
s_clf.fit(ft3, labels)

pickle.dump(s_clf, open("my_classifier.pkl", "w"))
pickle.dump(my_dataset, open("my_dataset.pkl", "w"))
pickle.dump(best_features, open("my_feature_list.pkl", "w"))
