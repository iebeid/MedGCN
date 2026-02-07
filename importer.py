import time
import re
import random
import chardet

from copy import deepcopy

import collections.abc
from collections import defaultdict
from collections import Counter
from collections import deque

from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx

import sklearn as skl
import tensorflow as tf

from sklearn.manifold import TSNE
from gensim.models import Word2Vec

from tree import Node, Tree

np.random.seed(123)
tf.random.set_seed(123)
tf.config.threading.set_inter_op_parallelism_threads(8)
np.set_printoptions(threshold=np.inf)


class Data:
    def __init__(self):
        pass


class Attribute(dict):
    def __init__(self):
        super().__init__()


class Shape(Attribute):
    def __init__(self, input, output, batch_size):
        super().__init__()

        self.input = input
        self.output = output

        self.batch_size = batch_size


class Hyperparameters(Attribute):
    def __init__(self):
        super().__init__()

    def add(self, key, value):
        self[key] = value


class Parameter:
    id = 0

    def __init__(self, shape: Shape, initializer: str = "glorot"):
        Parameter.id += 1
        self.label = "param_" + str(Parameter.id)
        self.shape = shape
        self.initializer = initializer


class Input(Attribute):
    id = 0

    def __init__(self, shape: Shape = None, input=None, initializer: str = "identity"):
        super().__init__()
        Input.id += 1

        self.label = "input_" + str(Input.id)

        self.input = input
        self.output = None
        self.shape = shape

        self.initializer = initializer

        if self.input:
            self.output = self.input
            self.shape = self.input.shape
        else:
            if self.initializer == "identity":
                self.output = UtilityOperation.identity_initializer(self.shape.batch_size,
                                                                    self.shape.output).numpy().tolist()
            elif self.initializer == "glorot":
                self.output = UtilityOperation.glorot_intializer(self.shape.batch_size,
                                                                 self.shape.output).numpy().tolist()


class ModelPointer:
    def __init__(self, input: Input, data: Data):
        self.input = input
        self.data = data


class Model(Tree):
    def __init__(self, root):
        super().__init__(root)

    def compute(self):
        pass


class Module:
    id = 0

    def __init__(self):
        # super().__init__(self, "parent")
        Module.id += 1
        self.label = "module_" + str(Module.id)

        # self.input = self.parent.parent.input
        # self.output = self.input.output
        # self.shape = self.input.shape

        self.input = None
        self.output = None
        self.shape = None

        self.parameters = []

    def get_parameters(self):
        self.parameters = self.get_leaves()

    def compute(self):
        pass


class Layer:
    id = 0

    def __init__(self, input: Input):
        super().__init__()
        Layer.id += 1
        self.label = "layer_" + str(Layer.id)

        self.input = input
        self.output = self.input.output
        self.shape = self.input.shape

        # self.input = None
        # self.output = None
        # self.shape = None

        self.embedding = None
        self.regularization_constant = None

        self.parameters = []
        # self.parameters[self.label] = {}

    def __str__(self):
        # return str(self.output)
        pass

    # def get_parameters(self):
    #     current_node = self.head
    #     while current_node is not None:
    #         if current_node.data.parameters:
    #             self.parameters[self.label][current_node.data.label] = current_node.data.get_parameters()[
    #                 current_node.data.label]
    #             current_node = current_node.next
    #         else:
    #             current_node = current_node.next
    #     return self.parameters
    #
    # def set_parameters(self, updated_parameters):
    #     current_node = self.head
    #     while current_node is not None:
    #         if current_node.data.parameters:
    #             current_node.data.set_parameters(updated_parameters)
    #             current_node = current_node.next
    #         else:
    #             current_node = current_node.next
    #
    # def compute(self):
    #     current_node = self.head
    #     result = self.output
    #     while current_node is not None:
    #         current_node.data.compute()
    #         if isinstance(current_node.data, Regularizer):
    #             self.regularization_constant = current_node.data.regularization_constant
    #         if isinstance(current_node.data, Normalizer):
    #             self.embedding = current_node.data.embedding
    #         result = current_node.data.output
    #         current_node = current_node.next
    #     self.output = result

    def compute(self):
        pass


class Operation:
    def __init__(self):
        self.parameters = {}

    def search_model(self, data):
        pass


class UtilityOperation(Operation):
    def __init__(self):
        super().__init__()

    # Algorithms
    @staticmethod
    def invert_dict(d):
        inverted_dict = {}
        for key, value_list in d.items():
            for value in value_list:
                inverted_dict.setdefault(value, set()).add(key)
        return inverted_dict

    @staticmethod
    def invert_dict_by_object_type(d):
        inverted_dict = {}
        model_objects = []
        module_objects = []
        layer_objects = []
        operation_objects = []
        parameter_objects = []
        for key, value in d.items():
            if isinstance(value, Model):
                # inverted_dict.setdefault(value, list()).append(key)
                model_objects.append(key)
                inverted_dict["Model"] = model_objects
            if isinstance(value, Module):
                # inverted_dict.setdefault(value, list()).append(key)
                module_objects.append(key)
                inverted_dict["Module"] = module_objects
            if isinstance(value, Layer):
                # inverted_dict.setdefault(value, list()).append(key)
                layer_objects.append(key)
                inverted_dict["Layer"] = layer_objects
            if isinstance(value, Operation):
                # inverted_dict.setdefault(value, list()).append(key)
                operation_objects.append(key)
                inverted_dict["Operation"] = operation_objects
            if isinstance(value, Parameter):
                # inverted_dict.setdefault(value, list()).append(key)
                parameter_objects.append(key)
                inverted_dict["Parameter"] = parameter_objects
        return inverted_dict

    @staticmethod
    def reverse_dict(d):
        reversed_dict = defaultdict(list)
        for key, value in d.items():
            reversed_dict[value].append(key)
        return reversed_dict

    @staticmethod
    def cosine_distance(vector1, vector2):
        return sp.spatial.distance.cosine(vector1, vector2)

    @staticmethod
    def flip_list(list):
        flipped_edges = []
        for es in list:
            flipped_edges.append((es[1], es[0], es[2]))
        return flipped_edges

    @staticmethod
    def determine_file_encoding(input_file_name):
        encoding = None
        with open(input_file_name, "rb") as rawdata:
            result = chardet.detect(rawdata.read())
            encoding = result["encoding"]
            if encoding == "ascii":
                encoding = "utf-8"
        return encoding

    @staticmethod
    def wl(graph1, graph2):
        # Generated by Gemini
        """Performs the Weisfeiler-Lehman test on two graphs.

        Args:
          graph1: The first graph.
          graph2: The second graph.

        Returns:
          True if the graphs are isomorphic, False otherwise.
        """

        # Initialize the node labels.
        node_labels1 = {node: node for node in graph1.nodes()}
        node_labels2 = {node: node for node in graph2.nodes()}

        # Iterate until the node labels converge.
        while True:
            # Update the node labels.
            for node in graph1.nodes():
                node_labels1[node] = (node_labels1[node],
                                      tuple(node_labels1[neighbor] for neighbor in graph1.neighbors(node)))
            for node in graph2.nodes():
                node_labels2[node] = (node_labels2[node],
                                      tuple(node_labels2[neighbor] for neighbor in graph2.neighbors(node)))

            # Check if the node labels have converged.
            if node_labels1 == node_labels2:
                return True

            # If the node labels have not converged, continue iterating.

        return False

    @staticmethod
    def are_matrices_identical(A, B):
        """ Generated by Gemini
        Compares two matrices and returns True if they are identical, False otherwise.

        Args:
          A: A list of lists representing the first matrix.
          B: A list of lists representing the second matrix.

        Returns:
          True if the two matrices are identical, False otherwise.
        """

        # Check if the dimensions of the two matrices are equal.
        if len(A) != len(B) or len(A[0]) != len(B[0]):
            return False

        # Compare each element of both matrices.
        for i in range(len(A)):
            for j in range(len(A[0])):
                if A[i][j] != B[i][j]:
                    return False

        # If all elements are equal, return True.
        return True

    @staticmethod
    def object_equality(o1, o2):
        if o1 == o2:
            return True
        else:
            return False

    @staticmethod
    def jaccard(list1, list2):
        # Gemini generated
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(set(list1)) + len(set(list2))) - intersection
        result = float(intersection) / union
        if result > 0.5:
            return True
        else:
            return False

    # Initializers
    @staticmethod
    def glorot_intializer(in_d, out_d):
        init = tf.keras.initializers.GlorotUniform()
        return init(shape=(in_d, out_d), dtype=tf.float32)

    @staticmethod
    def identity_initializer(in_d, out_d):
        init = tf.keras.initializers.Identity()
        return init(shape=(in_d, out_d), dtype=tf.float32)

    # Loss function
    @staticmethod
    def masked_cross_entropy_loss_evaluater(prediction, y, r, mask):
        return (-tf.reduce_mean(
            tf.reduce_sum(tf.boolean_mask(y, mask) * tf.math.log(tf.boolean_mask(prediction, mask))))) + r

    # Optimizers
    @staticmethod
    def optimizer(learn_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
        return optimizer

    # Evaluation from Kipf et al 2017
    @staticmethod
    def masked_accuracy_evaluater(prediction, y, mask):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        accuracy = tf.reduce_mean(accuracy_all)
        return accuracy
