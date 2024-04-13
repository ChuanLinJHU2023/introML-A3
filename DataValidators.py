# DataValidators.py implements some validators to test predictors, such as 5*2 cross validator.

import numpy as np
from DataPreprocessing import *
class KFoldCrossValidator:
    """Implements the K-fold cross validation"""
    k = None
    evaluation_metric = None
    stratify = None
    predictor = None

    def __init__(self, predictor, k=None, evaluation_metric=None, stratify=True):
        """Initialize the validator using certain predictor"""
        self.k = k
        self.evaluation_metric = evaluation_metric
        self.stratify = stratify
        self.predictor = predictor

    def validate(self, df_train, df_validation=None, show_detail=False):
        """Implement validation, including predictor training and testing"""
        ratio_list = [1] * self.k
        label_feature = self.predictor.get_label_feature() if self.stratify else None
        k_folds = data_partitioner(df_train, ratio_list, stratifyFeature=label_feature)
        k_experiment_results = []
        if show_detail:
            print("n of train dataset is {}".format(df_train.shape[0] - df_train.shape[0] // self.k))
            value = df_train.shape[0] // self.k if df_validation is None else df_validation.shape[0]
            print("n of test dataset is {}".format(value))
        for i in range(self.k):
            df_test = k_folds[i] if df_validation is None else df_validation
            df_train_small = pd.concat(k_folds[:i] + k_folds[i + 1:])
            self.predictor.fit(df_train_small,relearn_VDM=True)
            prediction = self.predictor.predict(df_test)
            label_feature = self.predictor.get_label_feature()
            metric_value = data_prediction_evaluator(df_test[label_feature], prediction, metric=self.evaluation_metric, show_more_detail=True)
            k_experiment_results.append(metric_value)
            if show_detail:
                print("in fold {}, we have {}={}".format(i, self.evaluation_metric, metric_value))
        return k_experiment_results


class KByTwoValidator:
    """Implements the K*2 cross validation"""
    k = None
    evaluation_metric = None
    stratify = None
    predictor = None
    usage = None

    def __init__(self, predictor, k=None, evaluation_metric=None, stratify=True):
        """Initialize the validator using certain predictor"""
        self.k = k
        self.evaluation_metric = evaluation_metric
        self.stratify = stratify
        self.predictor = predictor


    def validate(self, df_train, df_validation=None, show_detail=False, show_more_detail=False):
        """Implement validation, including predictor training and testing"""
        k_doubled_experiment_results = []
        for i in range(self.k):
            if show_detail:
                print("This is the {}th 2-fold cross validation".format(i))
            k = 2
            validator = KFoldCrossValidator(self.predictor, k, self.evaluation_metric, self.stratify)
            validation = validator.validate(df_train, df_validation=df_validation, show_detail=show_more_detail)
            k_doubled_experiment_results.extend(validation)
            if show_detail:
                print("the validation is {}\n".format(validation))
        if show_detail:
            print("the final results is:")
            print(k_doubled_experiment_results)
            print("the mean is:")
            print(np.mean(k_doubled_experiment_results))
        return k_doubled_experiment_results


class HyperParameterTuner:
    """Implements the hyperparameter tuning"""
    learner=None
    common_parameter_combination=None
    different_parameter_combinations=None

    def __init__(self, learner=None, common_parameter_combination=None, different_parameter_combinations=None):
        """Initialize the Tuner using certain predictor"""
        self.learner=learner
        self.common_parameter_combination=common_parameter_combination
        self.different_parameter_combinations=different_parameter_combinations

    def validate(self,df_train,df_validation,evaluation_metric,stratify=True,show_detial=False,show_more_detial=False):
        """Implement validation on different parameter combinations, including predictor training and testing"""
        validations=[]
        if show_detial:
            print("The common parameters are:")
            print(self.common_parameter_combination)
            print("\n\n")
        for parameter_combination in self.different_parameter_combinations:
            if show_detial:
                print("This is the 5*2 cross validation for:")
                print(parameter_combination)
            learner=self.learner
            predictor=learner(**self.common_parameter_combination,**parameter_combination)
            validator=KByTwoValidator(predictor, k=5, evaluation_metric=evaluation_metric, stratify=stratify)
            validation=validator.validate(df_train,show_detail=show_detial,show_more_detail=show_more_detial,df_validation=df_validation)
            validations.append(validation)
            if show_detial:
                print("\n\n")
        if show_detial:
            print("The final results is:")
            print(validations)
            print("their means are:")
            print([np.mean(v) for v in validations])
        return validations


class KFoldCrossValidatorMultiple:
    """Implements the K-fold cross validation"""
    k = None
    evaluation_metric = None
    stratify = None
    predictors = None
    n_of_predictors = None

    def __init__(self, predictors, k=None, evaluation_metric=None, stratify=True):
        """Initialize the validator using certain predictor"""
        assert isinstance(predictors,list)
        assert k
        assert evaluation_metric
        self.k = k
        self.evaluation_metric = evaluation_metric
        self.stratify = stratify
        self.predictors = predictors
        self.n_of_predictors = len(predictors)

    def validate(self, df_train, df_validation=None, show_detail=False):
        """Implement validation, including predictor training and testing"""
        ratio_list = [1] * self.k
        label_feature = self.predictors[0].get_label_feature() if self.stratify else None
        k_folds = data_partitioner(df_train, ratio_list, stratifyFeature=label_feature)
        k_experiment_results_for_all_predictors = [list() for i in range(self.n_of_predictors)]
        if show_detail:
            print("n of train dataset is {}".format(df_train.shape[0] - df_train.shape[0] // self.k))
            value = df_train.shape[0] // self.k if df_validation is None else df_validation.shape[0]
            print("n of test dataset is {}".format(value))
        for fold_index in range(self.k):
            df_test = k_folds[fold_index] if df_validation is None else df_validation
            df_train_small = pd.concat(k_folds[:fold_index] + k_folds[fold_index + 1:])
            for predictor_index in range(len(self.predictors)):
                predictor=self.predictors[predictor_index]
                metric_value=self.single_train_and_test(predictor,df_train_small,df_test)
                k_experiment_results_for_one_predictor=k_experiment_results_for_all_predictors[predictor_index]
                k_experiment_results_for_one_predictor.append(metric_value)
                if show_detail:
                    print("in fold {} for predictor {}, we have {}={}".format(fold_index, predictor_index, self.evaluation_metric, metric_value))
        return k_experiment_results_for_all_predictors

    def single_train_and_test(self,predictor,df_train_small,df_test):
        predictor.fit(df_train_small)
        prediction = predictor.predict(df_test)
        label_feature = predictor.get_label_feature()
        metric_value = data_prediction_evaluator(df_test[label_feature], prediction, metric=self.evaluation_metric)
        return metric_value


class KByTwoValidatorMultiple:
    """Implements the K*2 cross validation"""
    k = None
    evaluation_metric = None
    stratify = None
    predictors = None
    usage = None

    def __init__(self, predictors, k=None, evaluation_metric=None, stratify=True):
        """Initialize the validator using certain predictor"""
        assert isinstance(predictors,list)
        assert k>0
        assert evaluation_metric
        self.k = k
        self.evaluation_metric = evaluation_metric
        self.stratify = stratify
        self.predictors = predictors
        self.n_of_predictors = len(predictors)

    def validate(self, df_train, df_validation=None, show_detail=False, show_more_detail=False):
        """Implement validation, including predictor training and testing"""
        experiment_results_for_all_predictors = [list() for i in range(self.n_of_predictors)]
        for experiment_index in range(self.k):
            if show_detail:
                print("This is the {}th 2-fold cross validation".format(experiment_index))
            k = 2
            validator = KFoldCrossValidatorMultiple(self.predictors, k, self.evaluation_metric, self.stratify)
            validation = validator.validate(df_train, df_validation=df_validation, show_detail=show_more_detail)
            assert len(validation)==len(experiment_results_for_all_predictors)
            for predictor_index in range(self.n_of_predictors):
                experiment_results_for_all_predictors[predictor_index].extend(validation[predictor_index])
            if show_detail:
                for predictor_index in range(self.n_of_predictors):
                    print("the validation for {}th predictor is {}".format(predictor_index,validation[predictor_index]))
        if show_detail:
            for predictor_index in range(self.n_of_predictors):
                print("the final results for predictor {} is:".format(predictor_index))
                print(experiment_results_for_all_predictors[predictor_index])
                print("the mean is:")
                print(np.mean(experiment_results_for_all_predictors[predictor_index]))
        return experiment_results_for_all_predictors


