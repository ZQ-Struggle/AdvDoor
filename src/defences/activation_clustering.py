# -*- coding:utf-8 -*-

from sys import stdout
from analyzer import Analyzer
from conf import *
from poison_detection import ActivationDefence


class ActivationClustering:
    def __init__(self, data, param, model=None, activations=None):
        # model is activations when is_resume is True, and is DNN model when is_resume is False
        if model is None and activations is None:
            raise ("You must supply either model or activations")
        self.data = data
        self.param = param

        if activations is not None:
            self.defence = Analyzer(activations, self.data.x_train, self.data.y_train, self.param)
        else:
            self.model = model
            self.defence = ActivationDefence(self.model.classifier, self.data.x_train, self.data.y_train, self.param)

    def size_metric(self , log_file=None):
        # End-to-end method:
        print("------------------- Results using size metric -------------------")
        print(self.defence.get_params())
        self.defence.detect_poison(n_clusters=2, ndims=10, reduce="PCA")

        confusion_matrix = self.defence.evaluate_defence(self.data.is_clean)
        print("Evaluation defence results for size-based metric: ")
        jsonObject = json.loads(confusion_matrix)

     
        label = 'class_{}'.format(self.param.get_conf('poison_label_target'))
        print(label)
        pprint.pprint(jsonObject[label])

        self.print_f1_score(jsonObject, label)
        if log_file:
            savedStdout = sys.stdout
            sys.stdout = log_file
            for label in jsonObject:
                print(label)
                pprint.pprint(jsonObject[label])
                self.print_f1_score(jsonObject, label)
            sys.stdout= savedStdout
            

    def size_metric_visualize(self):
        # Visualize clusters:
        print("Visualize clusters")
        sprites_by_class = self.defence.visualize_clusters(self.data.x_train, 'mnist_poison_demo')
        # Show plots for clusters of class 5
        n_class = self.param.get_conf('poison_label_target')
        try:
            import matplotlib.pyplot as plt
            plt.imshow(sprites_by_class[n_class][0])
            plt.title("Class " + str(n_class) + " cluster: 0")
            plt.show()
            plt.imshow(sprites_by_class[n_class][1])
            plt.title("Class " + str(n_class) + " cluster: 1")
            plt.show()
        except:
            print("matplotlib not installed. For this reason, cluster visualization was not displayed")

    def distance_metric(self):
        print("------------------- Results using distance metric -------------------")
        print(self.defence.get_params())
        self.defence.detect_poison(n_clusters=2, ndims=10, reduce="PCA", cluster_analysis='distance')
        confusion_matrix = self.defence.evaluate_defence(self.data.is_clean)
        print("Evaluation defence results for distance-based metric: ")
        jsonObject = json.loads(confusion_matrix)

        # when result_dict is not empty, start record experiment results
        

        for label in jsonObject:
            print(label)
            pprint.pprint(jsonObject[label])

            self.print_f1_score(jsonObject, label)

        # Other ways to invoke the defence:
        self.defence.cluster_activations(n_clusters=2, ndims=10, reduce='PCA')

        self.defence.analyze_clusters(cluster_analysis='distance')
        self.defence.evaluate_defence(self.data.is_clean)

        self.defence.analyze_clusters(cluster_analysis='smaller')
        self.defence.evaluate_defence(self.data.is_clean)

    def relative_size_metric(self):
        print("------------------- Results using relative-size metric -------------------")
        print(self.defence.get_params())
        self.defence.detect_poison(n_clusters=2, ndims=10, reduce="PCA", cluster_analysis='relative-size')
        confusion_matrix = self.defence.evaluate_defence(self.data.is_clean)
        print("Evaluation defence results for relative-size metric: ")
        jsonObject = json.loads(confusion_matrix)

        # when result_dict is not empty, start record experiment results

        
        if type(self.param.get_conf('poison_label_target')) is list:
            for lab in self.param.get_conf('poison_label_target'):
                lab = 'class_{}'.format(lab)
                print(lab)
                pprint.pprint(jsonObject[lab])
                self.print_f1_score(jsonObject, lab)
        else:
            label = 'class_{}'.format(self.param.get_conf('poison_label_target'))
            print(label)
            pprint.pprint(jsonObject[label])
            self.print_f1_score(jsonObject, label)

    def print_f1_score(self, jsonObject, label):
        tp = jsonObject[label]['TruePositive']['numerator']
        fn = jsonObject[label]['FalseNegative']['numerator']
        tn = jsonObject[label]['TrueNegative']['numerator']
        fp = jsonObject[label]['FalsePositive']['numerator']

        if tp + fp == 0 or tp + fn == 0:
            print('escape the detection')
            return

        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
        if tp==0:
            f1 = 0.0
        else:
            f1 = (2 * precision * recall) / (precision + recall)

        print('precision = ', precision)
        print('recall = ', recall)
        print('f1 = ', f1)
