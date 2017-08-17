class model_selection(object):
    '''
    model selection 
    '''
    def __init__(self):
        lat_mean_scores = []
        lat_sem_scores  = []
        lat_best_scores  = []
        score_best      = 0
        clf_best        = []

        parameter_set = None
        train_test_split = None
        model = None
    
    def select(self, X, Y, model):
        for parameter in iter(parameter_set):
            lat_scores = []
            lat_best_score = 0
            for train_index, test_index in train_test_split.split(X, Y):
                # gather data
                X_train, Y_train = X[train_index, :], Y[train_index, :]
                X_test, Y_test = X[test_index, :], Y[test_index, :]
                model.fit(X_train, Y_train)
                true_scores = model.score(X_train, Y_train)
                pred_scores = model.score(X_test, Y_test)