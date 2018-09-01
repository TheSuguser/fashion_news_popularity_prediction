import numpy as np
from sklearn.naive_bayes import MultinomialNB


class DDM(object):

    def __init__(self, n, w, d,alpha, X_init, y_init, classes):
        """
        Initialize the drift detection method (DDM)

        Args:
            n: number of instance before starting to detect the concept drift
            w: number of standard deviation that indicates the warning level
            d: number of standard deviation that indicates the drift level
            alpha: the smoothing parameter of the multinomial Bayes classifier
            X_init: the training instances to initialize the base learner
            y_init: the labels of the instances that are used to initialize the base learner

        """
        if X_init.shape[0] != n:
            raise ValueError("The shape of X_init should be same as n")
        self.n = n
        self.w = w
        self.d = d
        self.p_min = 200
        self.s_min = 200
        self.p_com = -200
        self.store_token = None
        self.start_token = 0
        self.prob = []
        self.stat = []
        self.classes = classes
        self.X_train = X_init
        self.y_train = y_init
        self.alpha = alpha
        self.clf = MultinomialNB(alpha=alpha)
        self.clf.partial_fit(X_init, y_init, classes=classes)
        self.end_token = n+1

    def fit(self, X, y):
        """
        Fit the coming instance and detect the drift

        Args:
            X: training instance
            y: labels

        Returns:
            y_pred: the probability of the prediction 
            s: the state of this instance
               3 -> Concept drift; 
               2 -> False alarm; 
               1-> minimum; 
               0 -> Normal 
        """

        # Make the prediction and update the base learner
        self.X_train = np.concatenate(self.X_train, X, axis=0)
        self.y_train.append(y)
        y_pred = self.clf.predict_proba(X)
        self.clf.partial_fit(X, y, classes=self.classes)

        # Compute the error rate
        er = 1 - y_pred[0][y]
        std = np.sqrt((1-er)*er/(self.end_token - self.start_token))

        s = 0 # State token

        # Detect the drift
        # Minimum
        if er < self.p_min:
            self.p_min = er
            self.s_min = std
            s = 0

        # False alarm
        if er + std >= self.p_min + self.w * self.s_min:
            self.store_token = None
            self.p_con = -100
            s = 2

        # Concept drift
        if er + std >= self.p_min + self.d * self.s_min:
            if self.store_token == None:
                self.start_token = self.end_token - self.n

            else:
                self.start_token = self.store_token
                if self.end_token - self.store_token < self.n:
                    self.start_token = self.end_token - self.n

            self.p_min = 200
            self.s_min = 200
            s = 3

            self.clf = MultinomialNB(alpha = self.alpha)
            self.clf.partial_fit(self.X_trian[self.start_token: self.end_token,:],
                self.y_train[self.start_token:self.end_token],
                classes=self.classes)

        self.prob.append(y_pred)
        self.stat.append(s)

        return y_pred, s











