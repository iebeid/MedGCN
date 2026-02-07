from importer import *
from grapher import *


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def add(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node

    def print_list(self):
        current_node = self.head
        while current_node is not None:
            print(current_node.data)
            current_node = current_node.next


class TrainableOperation(Operation):
    def __init__(self):
        super().__init__()


class LayerOperation(Operation):
    def __init__(self):
        super().__init__()


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


class Input(Attribute):
    def __init__(self, id: int, shape: Shape = None, input=None, initializer: str = "identity"):
        super().__init__()
        self.label = "input_" + str(id)

        self.input = input
        self.output = None
        self.shape = shape

        self.initializer = initializer

        if self.input:
            self.output = self.input
            # self.parameters = self.input.parameters
            self.parameters = {}
            self.parameters[self.label] = {"W1": [], "b1": []}
            self.shape = self.input.shape
        else:
            if self.initializer == "identity":
                self.output = UtilityOperation.identity_initializer(self.shape.batch_size,
                                                                    self.shape.output).numpy().tolist()
            elif self.initializer == "glorot":
                self.output = UtilityOperation.glorot_intializer(self.shape.batch_size,
                                                                 self.shape.output).numpy().tolist()





class Layer(DoublyLinkedList):
    def __init__(self, id: int, input: Input):
        super().__init__()
        self.label = "layer_" + str(id)

        self.input = input
        self.output = self.input.output
        self.shape = self.input.shape

        self.embedding = None
        self.regularization_constant = None

        self.parameters = {}
        self.parameters[self.label] = {}

    def __str__(self):
        return str(self.output)

    def get_parameters(self):
        current_node = self.head
        while current_node is not None:
            if current_node.data.parameters:
                self.parameters[self.label][current_node.data.label] = current_node.data.get_parameters()[
                    current_node.data.label]
                current_node = current_node.next
            else:
                current_node = current_node.next
        return self.parameters

    def set_parameters(self, updated_parameters):
        current_node = self.head
        while current_node is not None:
            if current_node.data.parameters:
                current_node.data.set_parameters(updated_parameters)
                current_node = current_node.next
            else:
                current_node = current_node.next

    def compute(self):
        current_node = self.head
        result = self.output
        while current_node is not None:
            current_node.data.compute()
            if isinstance(current_node.data, Regularizer):
                self.regularization_constant = current_node.data.regularization_constant
            if isinstance(current_node.data, Normalizer):
                self.embedding = current_node.data.embedding
            result = current_node.data.output
            current_node = current_node.next
        self.output = result


class Dropout(LayerOperation):
    def __init__(self, id: int, input: Input, rate: float):
        super().__init__()
        self.label = "drop_" + str(id)
        self.input = input
        self.output = None
        self.shape = self.input.shape

        # self.parameters = self.input.parameters
        # self.parameters = {"W1":[], "b1":[]}
        self.parameters = {}
        self.parameters[self.label] = {"W1": [], "b1": []}

        self.rate = rate

    def get_parameters(self):
        return self.parameters

    def compute(self):
        self.output = tf.nn.dropout(self.input.output, self.rate)


class Regularizer(LayerOperation):
    def __init__(self, id: int, input: Input, rate: float):
        super().__init__()
        self.label = "reg_" + str(id)
        self.input = input
        self.output = None
        self.shape = self.input.shape

        # self.parameters = self.input.parameters
        # self.parameters = {"W1":[], "b1":[]}
        self.parameters = {}
        self.parameters[self.label] = {"W1": [], "b1": []}

        self.rate = rate
        self.regularization_constant = None

    def get_parameters(self):
        return self.parameters

    def compute(self):
        r = tf.keras.regularizers.L2(self.rate)
        self.regularization_constant = r(self.input.output)
        self.output = self.input.output


class Normalizer(LayerOperation):
    def __init__(self, id: int, input: Input):
        super().__init__()
        self.label = "norm_" + str(id)
        self.input = input
        self.output = None
        self.shape = self.input.shape

        # self.parameters = self.input.parameters
        # self.parameters = {"W1":[], "b1":[]}
        self.parameters = {}
        self.parameters[self.label] = {"W1": [], "b1": []}

        self.embedding = None

    def get_parameters(self):
        return self.parameters

    def compute(self):
        self.embedding = tf.nn.l2_normalize(self.input.output, axis=1)
        self.output = self.input.output


class Activation(LayerOperation):
    def __init__(self, input: Input, shape: Attribute):
        super().__init__()
        self.input = input
        self.shape = shape


class TanhActivation(Activation):
    def __init__(self, id: int, input: Input, shape: Shape):
        super().__init__(input, shape)
        self.label = "tanh_" + str(id)

        self.input = input
        self.output = None
        self.shape = shape

        # self.parameters = self.input.parameters
        self.parameters = {}
        self.parameters[self.label] = {"W1": [], "b1": []}
        # self.parameters =

    def get_parameters(self):
        return self.parameters

    def compute(self):
        self.output = tf.nn.tanh(self.input.output)


class SoftmaxActivation(Activation):
    def __init__(self, id: int, input: Input, shape: Attribute):
        super().__init__(input, shape)
        self.label = "softmax_" + str(id)
        self.input = input
        self.output = None
        self.shape = shape

        self.parameters = self.input.parameters

    def compute(self):
        self.output = tf.nn.softmax(self.input.output)


class ReluActivation(Activation):
    def __init__(self, id: int, input: Input, shape: Attribute):
        super().__init__(input, shape)
        self.label = "relu_" + str(id)
        self.input = input
        self.output = None
        self.shape = shape

        self.parameters = self.input.parameters

    def compute(self):
        self.output = tf.nn.relu(self.input.output)


class SoftmaxPrediction(LayerOperation):
    def __init__(self, id: int, input: Input, shape: Attribute):
        super().__init__()
        self.label = "softmax_prediction_" + str(id)
        self.input = input
        self.output = None
        self.shape = shape

        # self.parameters = self.input.parameters
        self.parameters = {}
        self.parameters[self.label] = {"W1": [], "b1": []}

    def get_parameters(self):
        return self.parameters

    def compute(self):
        self.output = tf.map_fn(fn=lambda i: tf.nn.softmax(i), elems=self.input.output)


class BatchNormalization(TrainableOperation):
    def __init__(self, id: int, input: Input, shape: Shape, factor=0.001):
        super().__init__()

        self.label = "bn_" + str(id)

        self.input = input
        self.output = None
        self.shape = shape

        self.factor = factor

        self.parameters = {}
        self.parameters[self.label] = {}
        self.parameters[self.label]["W1"] = tf.Variable(
            UtilityOperation.glorot_intializer(1, self.shape.output)).numpy().tolist()
        self.parameters[self.label]["b1"] = tf.Variable(
            UtilityOperation.glorot_intializer(1, self.shape.output)).numpy().tolist()

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, updated_parameters):
        self.parameters[self.label]["W1"] = updated_parameters[self.label]["W1"]
        self.parameters[self.label]["b1"] = updated_parameters[self.label]["b1"]

    def compute(self):
        batch_mean, batch_var = tf.nn.moments(self.input.output, [0])
        self.output = tf.nn.sigmoid(
            tf.Variable(tf.convert_to_tensor(self.parameters[self.label]["W1"])) * (
                    (self.input.output - batch_mean) / tf.sqrt(
                batch_var + self.factor)) + tf.Variable(tf.convert_to_tensor(self.parameters[self.label]["b1"])))


class Residual(TrainableOperation):
    def __init__(self, id: int, input1: Input, input2: Input, shape: Shape):
        super().__init__()

        self.label = "res_" + str(id)

        self.input1 = input1
        self.input2 = input2
        self.output = None
        self.shape = shape

        self.parameters = {}
        self.parameters[self.label] = {}
        self.parameters[self.label]["W1"] = tf.Variable(
            UtilityOperation.glorot_intializer(self.shape.input, self.shape.output)).numpy().tolist()
        self.parameters[self.label]["b1"] = tf.Variable(
            UtilityOperation.glorot_intializer(self.shape.batch_size, self.shape.output)).numpy().tolist()

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, updated_parameters):
        self.parameters[self.label]["W1"] = updated_parameters[self.label]["W1"]
        self.parameters[self.label]["b1"] = updated_parameters[self.label]["b1"]

    def compute(self):
        self.output = tf.add(self.input2.output,
                             tf.add(tf.matmul(self.input1.output,
                                              tf.Variable(tf.convert_to_tensor(self.parameters[self.label]["W1"]))),
                                    tf.Variable(tf.convert_to_tensor(self.parameters[self.label]["b1"]))))


class GraphConvolution(TrainableOperation):
    def __init__(self, id: int, input: Input, graph: Graph, shape: Shape):
        super().__init__()
        self.label = "gcn_" + str(id)

        self.input = input
        self.output = None
        self.shape = shape

        self.graph = graph

        self.parameters = {}
        self.parameters[self.label] = {}
        self.parameters[self.label]["W1"] = tf.Variable(
            UtilityOperation.glorot_intializer(self.shape.input, self.shape.output)).numpy().tolist()
        self.parameters[self.label]["b1"] = tf.Variable(
            UtilityOperation.glorot_intializer(self.shape.batch_size, self.shape.output)).numpy().tolist()

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, updated_parameters):
        self.parameters[self.label]["W1"] = updated_parameters[self.label]["W1"]
        self.parameters[self.label]["b1"] = updated_parameters[self.label]["b1"]

    def compute(self):
        self.output = tf.add(tf.matmul(
            tf.matmul(tf.convert_to_tensor(self.graph.degree_normalized_adjacency, dtype=tf.float32),
                      self.input.output
                      ), tf.Variable(tf.convert_to_tensor(self.parameters[self.label]["W1"]))),
            tf.Variable(tf.convert_to_tensor(self.parameters[self.label]["b1"])))


class DirectedGraphConvolution(TrainableOperation):
    def __init__(self, id: int, input: Input, graph: Graph, shape: Shape):
        super().__init__()
        self.label = "dgcn_" + str(id)

        self.input = input
        self.output = None
        self.shape = shape

        self.graph = graph

        self.parameters = {}
        self.parameters[self.label] = {}
        self.parameters[self.label]["W1"] = tf.Variable(
            UtilityOperation.glorot_intializer(self.shape.input, self.shape.output))
        self.parameters[self.label]["b1"] = tf.Variable(
            UtilityOperation.glorot_intializer(self.shape.batch_size, self.shape.output))
        self.parameters[self.label]["W2"] = tf.Variable(
            UtilityOperation.glorot_intializer(self.shape.input, self.shape.output))
        self.parameters[self.label]["b2"] = tf.Variable(
            UtilityOperation.glorot_intializer(self.shape.batch_size, self.shape.output))

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, updated_parameters):
        self.parameters[self.label]["W1"] = updated_parameters[self.label]["W1"]
        self.parameters[self.label]["b1"] = updated_parameters[self.label]["b1"]
        self.parameters[self.label]["W2"] = updated_parameters[self.label]["W2"]
        self.parameters[self.label]["b2"] = updated_parameters[self.label]["b2"]

    def compute(self):
        a = tf.add(tf.matmul(
            tf.matmul(tf.convert_to_tensor(self.graph.degree_normalized_weighted_out_adjacency, dtype=tf.float32),
                      self.input.output
                      ), self.parameters[self.label]["W2"]), self.parameters[self.label]["b2"])
        aT = tf.transpose(a)
        b = tf.add(tf.matmul(
            tf.matmul(tf.convert_to_tensor(self.graph.degree_normalized_weighted_in_adjacency, dtype=tf.float32),
                      self.input.output
                      ), self.parameters[self.label]["W1"]), self.parameters[self.label]["b1"])
        bT = tf.transpose(b)
        self.output = tf.add_n([tf.matmul(a, aT), tf.matmul(a, bT), tf.matmul(b, aT), tf.matmul(b, bT)])


class Dense(TrainableOperation):
    def __init__(self, id: int, input: Input, shape: Attribute):
        super().__init__()
        self.label = "dense_" + str(id)
        self.input = input
        self.output = None
        self.shape = shape

        self.parameters = {}
        self.parameters[self.label] = {}
        self.parameters[self.label]["W1"] = tf.Variable(
            UtilityOperation.glorot_intializer(self.shape.input, self.shape.output)).numpy().tolist()
        self.parameters[self.label]["b1"] = tf.Variable(
            UtilityOperation.glorot_intializer(self.shape.batch_size, self.shape.output)).numpy().tolist()

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, updated_parameters):
        self.parameters[self.label]["W1"] = updated_parameters[self.label]["W1"]
        self.parameters[self.label]["b1"] = updated_parameters[self.label]["b1"]

    def compute(self):
        self.output = tf.add(
            tf.matmul(self.input.output, tf.Variable(tf.convert_to_tensor(self.parameters[self.label]["W1"]))),
            tf.Variable(tf.convert_to_tensor(self.parameters[self.label]["b1"])))
