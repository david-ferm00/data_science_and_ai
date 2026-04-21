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

def train_test_split_manual(X, y):
    """
    Splits the arrays X and y into random train and test subsets.

    Parameters:
    - X : the n*d matrix of observations
    - y : the n-vector of class labels
    - train_size : the fraction of rows to use for training
    - random_state : a random seed, or None for unpredictable random initialization
    """
    assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0]

    random_state = 123456
    train_size = 0.75
    take_order = [i for i in range(len(x)-1)]
    np.random.default_rng(seed=random_state).shuffle(take_order)
    X_train = x[take_order[:int(train_size*len(x))]]
    y_train = y[take_order[:int(train_size*len(y))]]
    X_test = x[take_order[int(train_size*len(x)):]]
    y_test = y[take_order[int(train_size*len(y)):]]

    return (X_train, X_test, y_train, y_test)



class NaiveBinaryBayes:
    """
    Naive Bayes classifier for binary features and binary class labels.

    Matches the interface of sklearn.naive_bayes.CategoricalNB
    """

    n_features_in = None
    classes_ = None
    class_count_ = None
    class_log_prior_ = None
    category_count_ = []
    feature_log_prob_ = []
    alpha = None

    
    def __init__(self, alpha = 1.0):
        """
        Construct the classifier object.

        Parameters:
        - alpha : Parameter for Laplace smoothing (how many pseudoinstances to add for each category)
        """
        self.alpha = alpha

    

    
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

        self.n_features_in = X.shape[1]
        self.classes_ = np.array([0, 1])
        self.class_count_ = np.array([n0, n1])
        self.class_log_prior_ = np.log(self.class_count_/len(y))

        for i in range(self.n_features_in):
            temp_matrix_category = [[0,0],
                                    [0,0]]
            temp_matrix_log_prob = [[0,0],
                                    [0,0]]
            for j in range(len(y)):
                first_index  = int(y[j])
                second_index = int(X[j][i])
                temp_matrix_category[first_index][second_index] += 1
            
            

            self.category_count_.append(np.array(temp_matrix_category))
            #self.feature_log_prob_.append(np.array(temp_matrix_log_prob))

        for f in range(self.n_features_in):
            m = 2
            counts = self.category_count_[f].astype(np.float64)

            probs = (self.alpha + counts) / (self.class_count_[:, None] + self.alpha*m) 

            log_probs = np.log(probs)

            self.feature_log_prob_.append(log_probs)

        return self

    
    def predict_log_proba(self, X):
        """
        Given an m*d array X, returns an m*2 array of log probabilities corresponding to the posterior probability of the observation coming from a given class

        Parameters:
        - X: an m*d array of observations to predict

        Return value:
        - An m*2 array of log probabilities
        """
        assert X.ndim == 2 and X.shape[1] == self.n_features_in
        
        result = []

        for m in X:
            have = 0
            not_have = 0
            for feature in self.feature_log_prob_:
                second_index = int(m[self.feature_log_prob_.index(feature)])
                have += feature[1][second_index] 
                not_have += feature[0][second_index]
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
        assert X.ndim == 2 and X.shape[1] == self.n_features_in

        result = []

        for row in self.predict_log_proba(X):
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
        assert X.ndim == 2 and X.shape[1] == self.n_features_in

        result = []
        for row in self.predict_proba(X):
            result.append(np.argmax(row))

        return np.array(result)


def confusion_matrix_manual(y_true, y_pred):
    """
    Computes a confusion matrix for binary classification predictions.

    Parameters:
    - y_true: vector, ground truth
    - y_pred: vector, predictions

    Return value:
    - A 2*2 matrix of classification values
    """
    result = [[0,0], [0,0]]
    for i in range(2):
        for j in range(2):
            for k in range(len(y_true)):
                if y_true[k] == i and y_pred[k] == j:
                    result[i][j] += 1
        

    return result

def accuracy_score_manual(confusion_matrix):
    """
    Given confusion matrix, computes the accuracy

    Parameters:
    - confusion_matrix: a 2*2 matrix

    Return value:
    - floating point, accuracy (fraction of correct classifications)
    """
    return ((confusion_matrix[0][0] + confusion_matrix[1][1]) / np.sum(confusion_matrix))
    
def precision_score_manual(confusion_matrix):
    """
    Given confusion matrix, computes the precision

    Parameters:
    - confusion_matrix: a 2*2 matrix

    Return value:
    - floating point, precision (fraction of relevant entries among positive classifications)
    """
    return (confusion_matrix[1][1]/ (confusion_matrix[1][1] + confusion_matrix[0][1]))

def recall_score_manual(confusion_matrix):
    """
    Given confusion matrix, computes the recall

    Parameters:
    - confusion_matrix: a 2*2 matrix

    Return value:
    - floating point, recall (fraction of relevant entries returned)
    """
    return (confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0]))
    
def f1_score_manual(confusion_matrix):
    """
    Given confusion matrix, computes the F-score

    Parameters:
    - confusion_matrix: a 2*2 matrix

    Return value:
    - floating point, F-score (harmonic mean of precision and recall)
    """

    return (2/((1/precision_score_manual(confusion_matrix)) + (1/recall_score_manual(confusion_matrix))))
    

df = load_binary_data("diabetes_binary_5050split_health_indicators_BRFSS2015-1.csv")
y = np.array(df[binary_target])
x = np.array(df.copy().drop(binary_target, axis=1))

X_train, X_test, y_train, y_test = train_test_split_manual(x, y)

model = NaiveBinaryBayes(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

confusion_matrix = confusion_matrix_manual(y_test, y_pred)
accuracy = accuracy_score_manual(confusion_matrix)
precision = precision_score_manual(confusion_matrix)
recall = recall_score_manual(confusion_matrix)
f1 = f1_score_manual(confusion_matrix)

print("Confusion Matrix: " + str(confusion_matrix))
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1 Score: " + str(f1))












































