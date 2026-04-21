##csv into datafram with only binaries
## name of the target variable in "binary_target"
##names of columns that are binary in "binary_features"
import pandas as pd
import numpy as np

binary_features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']
binary_target = 'Diabetes_binary'
def load_binary_data(filename):
    """Load a CSV file into a Pandas dataframe, keeping only the binary features and the target variable.

    Parameters:
    filename -- name of the CSV file to load
    binary_target -- name of the target variable (column) in the CSV file
    binary_features -- list of names of the binary feature columns in the CSV file
    """
    df = pd.read_csv(filename)
    for column in df.columns:
        if column not in binary_features and column != binary_target:
            df.drop(column, axis=1, inplace=True)
    return df   

df = load_binary_data("diabetes_binary_5050split_health_indicators_BRFSS2015-1.csv")

y = np.array(df[binary_target])
x = np.array(df.copy().drop(binary_target, axis=1))

#########
def train_test_split_manual(x, y):
    random_state = 123456
    train_size = 0.75
    take_order = [i for i in range(len(x)-1)]
    np.random.default_rng(seed=random_state).shuffle(take_order)
    X_train = x[take_order[:int(train_size*len(x))]]
    y_train = y[take_order[:int(train_size*len(y))]]
    X_test = x[take_order[int(train_size*len(x)):]]
    y_test = y[take_order[int(train_size*len(y)):]]

    return (X_train, X_test, y_train, y_test)

train_test_split_manual(x, y)

#########
def fit(self,X,y):
    """
    Fit the dataset X with correct class labels y into the classfier.

    Defines the following member variables:
    - n_feature_in : integer, the number of features d
    - classes_ : array, class labels
    - class_count_ : array, count of classes encountered during training
    - class_log_prior_ : logarithm of empirical prior (ln (class count / n) for each class)
    - category_count_: a list of length d, one array per feature; each array is of shape 2*2 and the element at index (j,k) in the ith array corresponds to the number of observations coming from class j that have value k for the ith feature 
    - feature_log_prob_: a list of length d, one array per feature; the element corresponds to the log of matching category count divided by the class count

    Parameters:
    - X : n observations (rows) of d features (columns), all must be binary
    - y : n class labels (binary)

    Return value:
    - Returns self
    """
    assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0]
    
    n0 = 0
    n1 = 0
    for row in y:
        if row == 0:
            n0 += 1
        elif row == 1:
            n1 += 1
        else:
            raise ValueError("y must be binary")

    self.n_feature_in = X.shape[1]
    self.classes_ = np.array([0, 1])
    self.class_count_ = np.array([n0, n1])
    self.class_log_prior_ = np.array([np.log(self.class_count_[0] / len(y)), np.log(self.class_count_[1] / len(y))])

    for i in range(self.n_feature_in):
        temp_matrix_category = [[[],[]],
                                [[],[]]]
        temp_matrix_log_prob = [[[],[]],
                                [[],[]]]
        for j in range(len(y)):
            temp_matrix_category[y[j]][X[y[j]][i]] += 1
        
        for j in temp_matrix_category:
            for k in j:
                temp_matrix_log_prob[temp_matrix_category.index(j)][temp_matrix_category.index(k)] = np.log((self.alpha + k) / (self.alpha * 2 + len(y)))

        self.category_count_.append(np.array(temp_matrix_category))
        self.feature_log_prob_.append(np.array(temp_matrix_log_prob))


    return self


#######

def predict_log_proba(self, X):
    """
    Given an m*d array X, returns an m*2 array of log probabilities corresponding to the posterior probability of the observation coming from a given class

    Parameters:
    - X: an m*d array of observations to predict

    Return value:
    - An m*2 array of log probabilities
    """
    assert X.ndim == 2 and X.shape[1] == self.n_features_in_
    
    result = []

    for m in X:
        have = 0
        not_have = 0
        for feature in self.feature_log_prob_:
            have += feature[1][m[self.feature_log_prob_.index(feature)]] 
            not_have += feature[0][m[self.feature_log_prob_.index(feature)]]
        ##TODO do we even need to normalize here? we do it in predict_proba
        result.append([not_have, have])

    return np.array(result)

def predict_proba(self, X):
    """
    Given an m*d array X, returns an m*2 array of probabilities corresponding to the posterior probability of the observation coming from a given class

    Parameters:
    - X: an m*d array of observations to predict

    Return value:
    - An m*2 array of probabilities
    """
    assert X.ndim == 2 and X.shape[1] == self.n_features_in_

    result = []

    for row in self.predict_log_proba(self, X):
        not_have = np.exp(row[0] - np.max(row))
        have = np.exp(row[1] - np.max(row))

        total = not_have + have

        not_have = not_have / total
        have = have / total

        result.append([not_have, have])

    return np.array(result)

def predict(self, X):
    """
    Given an m*d array X, returns an m vector of predicted class labels

    Parameters:
    - X: an m*d array of observations to predict

    Return value:
    - An m array of class labels (corresponding to the maximum probability for each observation)
    """
    assert X.ndim == 2 and X.shape[1] == self.n_features_in_

    result = []

    for row in self.predict_proba(self, X):
        result.append(np.argmax(row))

    return np.array(result)

#########

def __init__(self, alpha = 1.0):
        """
        Construct the classifier object.

        Parameters:
        - alpha : Parameter for Laplace smoothing (how many pseudoinstances to add for each category)
        """
        self.alpha = alpha

#######


