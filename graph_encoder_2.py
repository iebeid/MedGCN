from importer import *
from grapher import *


class TrainableOperation(Operation):
    def __init__(self):
        super().__init__()


class LayerOperation(Operation):
    def __init__(self):
        super().__init__()


class Dropout(LayerOperation):
    id = 0

    def __init__(self, rate: float, shape: Shape):
        super().__init__()
        Dropout.id += 1
        self.label = "drop_" + str(Dropout.id)

        # self.input = input
        # self.output = None
        # self.shape = self.input.shape

        self.input = None
        self.output = None
        self.shape = shape

        # self.parameters = self.input.parameters
        self.parameters = []

        # self.parameters = {}
        # self.parameters[self.label] = {"W1": None, "b1": None}

        self.rate = rate

    # def get_parameters(self):
    #     return self.parameters

    def compute(self):
        self.output = tf.nn.dropout(self.input.output, self.rate)


class Regularizer(LayerOperation):
    id = 0

    def __init__(self, rate: float, shape: Shape):
        super().__init__()
        Regularizer.id += 1
        self.label = "regularizer_" + str(Regularizer.id)

        # self.input = input
        # self.output = None
        # self.shape = shape

        self.input = None
        self.output = None
        self.shape = shape

        # self.parameters = self.input.parameters
        # self.parameters = {"W1":[], "b1":[]}

        self.parameters = []

        self.rate = rate
        self.regularization_constant = None

    # def get_parameters(self):
    #     return self.parameters

    def compute(self):
        r = tf.keras.regularizers.L2(self.rate)
        self.regularization_constant = r(self.input.output)
        self.output = self.input.output


class Normalizer(LayerOperation):
    id = 0

    def __init__(self, shape: Shape):
        super().__init__()
        Normalizer.id += 1
        self.label = "normalizer_" + str(Normalizer.id)

        # self.input = input
        # self.output = None
        # self.shape = self.input.shape

        self.input = None
        self.output = None
        self.shape = shape

        # self.parameters = self.input.parameters
        # self.parameters = {"W1":[], "b1":[]}

        self.parameters = []

        self.embedding = None

    # def get_parameters(self):
    #     return self.parameters

    def compute(self):
        self.embedding = tf.nn.l2_normalize(self.input.output, axis=1)
        self.output = self.input.output


class Activation(LayerOperation):
    def __init__(self, shape: Attribute):
        super().__init__()

        # self.input = input
        self.input = None
        self.shape = shape


class TanhActivation(Activation):
    id = 0

    def __init__(self, shape: Shape):
        super().__init__(shape)
        TanhActivation.id += 1
        self.label = "tanh_" + str(TanhActivation.id)

        # self.input = input
        # self.output = None
        # self.shape = shape

        self.input = None
        self.output = None
        self.shape = shape

        # self.parameters = self.input.parameters

        self.parameters = []
        # self.parameters =

    # def get_parameters(self):
    #     return self.parameters

    def compute(self):
        self.output = tf.nn.tanh(self.input.output)


class SoftmaxActivation(Activation):
    id = 0

    def __init__(self, shape: Shape):
        super().__init__(input, shape)
        SoftmaxActivation.id += 1
        self.label = "softmax_" + str(SoftmaxActivation.id)

        # self.input = input
        # self.output = None
        # self.shape = shape

        self.input = None
        self.output = None
        self.shape = shape

        self.parameters = []

    def compute(self):
        self.output = tf.nn.softmax(self.input.output)


class ReluActivation(Activation):
    id = 0

    def __init__(self, shape: Shape):
        super().__init__(input, shape)
        ReluActivation.id += 1
        self.label = "relu_" + str(ReluActivation.id)

        # self.input = input
        # self.output = None
        # self.shape = shape

        self.input = None
        self.output = None
        self.shape = shape

        self.parameters = []

    def compute(self):
        self.output = tf.nn.relu(self.input.output)


class SoftmaxPrediction(LayerOperation):
    id = 0

    def __init__(self, shape: Shape):
        super().__init__()
        SoftmaxPrediction.id += 1
        self.label = "softmax_prediction_" + str(SoftmaxPrediction.id)

        # self.input = input
        # self.output = None
        # self.shape = shape

        self.input = None
        self.output = None
        self.shape = shape

        # self.parameters = self.input.parameters

        self.parameters = []

    # def get_parameters(self):
    #     return self.parameters

    def compute(self):
        self.output = tf.map_fn(fn=lambda i: tf.nn.softmax(i), elems=self.input.output)


class BatchNormalization(TrainableOperation):
    id = 0

    def __init__(self, shape: Shape, bn_factor=0.001):
        super().__init__()
        BatchNormalization.id += 1
        self.label = "batch_normalization_" + str(BatchNormalization.id)

        # self.input = input
        # self.output = None
        # self.shape = shape

        self.input = None
        self.output = None
        self.shape = shape

        self.bn_factor = bn_factor

        self.parameters = [Parameter(Shape(1, self.shape.output, self.shape.batch_size)),
                           Parameter(Shape(1, self.shape.output, self.shape.batch_size))]

    # def get_parameters(self):
    #     return self

    # def set_parameters(self, updated_parameters):
    #     self.parameters[self.label]["W1"] = updated_parameters[self.label]["W1"]
    #     self.parameters[self.label]["b1"] = updated_parameters[self.label]["b1"]

    def compute(self):
        batch_mean, batch_var = tf.nn.moments(self.input.output, [0])
        self.output = tf.nn.sigmoid(
            tf.Variable(tf.convert_to_tensor(parameter_bus[self.parameters[0].id])) * (
                    (self.input.output - batch_mean) / tf.sqrt(
                batch_var + self.bn_factor)) + tf.Variable(tf.convert_to_tensor(parameter_bus[self.parameters[1].id])))


class Residual(TrainableOperation):
    id = 0

    def __init__(self, shape: Shape):
        super().__init__()
        Residual.id += 1
        self.label = "residual_" + str(Residual.id)

        # self.input1 = input1
        # self.input2 = input2
        # self.output = None
        # self.shape = shape

        self.input1 = None
        self.input2 = None
        self.output = None
        self.shape = shape

        # self.parameters = {}
        # self.parameters[self.label] = {}
        # self.parameters[self.label]["W1"] = tf.Variable(
        #     UtilityOperation.glorot_intializer(self.shape.input, self.shape.output)).numpy().tolist()
        # self.parameters[self.label]["b1"] = tf.Variable(
        #     UtilityOperation.glorot_intializer(self.shape.batch_size, self.shape.output)).numpy().tolist()

        self.parameters = [Parameter(Shape(self.shape.input, self.shape.output, self.shape.batch_size)),
                           Parameter(Shape(self.shape.batch_size, self.shape.output, self.shape.batch_size))]

    # def get_parameters(self):
    #     return self.parameters

    # def set_parameters(self, updated_parameters):
    #     self.parameters[self.label]["W1"] = updated_parameters[self.label]["W1"]
    #     self.parameters[self.label]["b1"] = updated_parameters[self.label]["b1"]

    def compute(self):
        self.output = tf.add(self.input2.output,
                             tf.add(tf.matmul(self.input1.output,
                                              tf.Variable(tf.convert_to_tensor(parameter_bus[self.parameters[0].id]))),
                                    tf.Variable(tf.convert_to_tensor(parameter_bus[self.parameters[1].id]))))


class GraphConvolution(TrainableOperation):
    id = 0

    def __init__(self, shape: Shape):
        super().__init__()
        GraphConvolution.id += 1
        self.label = "gcn_" + str(GraphConvolution.id)

        # self.input = input
        # self.output = None
        # self.shape = shape
        #
        # self.graph = graph

        self.input = None
        self.output = None
        self.shape = shape

        self.graph = None

        # self.parameters = {}
        # self.parameters[self.label] = {}
        # self.parameters[self.label]["W1"] = tf.Variable(
        #     UtilityOperation.glorot_intializer(self.shape.input, self.shape.output)).numpy().tolist()
        # self.parameters[self.label]["b1"] = tf.Variable(
        #     UtilityOperation.glorot_intializer(self.shape.batch_size, self.shape.output)).numpy().tolist()

        self.parameters = [Parameter(Shape(self.shape.input, self.shape.output, self.shape.batch_size)),
                           Parameter(Shape(self.shape.batch_size, self.shape.output, self.shape.batch_size))]

    # def get_parameters(self):
    #     return self.parameters
    #
    # def set_parameters(self, updated_parameters):
    #     self.parameters[self.label]["W1"] = updated_parameters[self.label]["W1"]
    #     self.parameters[self.label]["b1"] = updated_parameters[self.label]["b1"]

    def compute(self):
        self.output = tf.add(tf.matmul(
            tf.matmul(tf.convert_to_tensor(self.graph.degree_normalized_adjacency, dtype=tf.float32),
                      self.input.output
                      ), tf.Variable(tf.convert_to_tensor(parameter_bus[self.parameters[0].id]))),
            tf.Variable(tf.convert_to_tensor(parameter_bus[self.parameters[1].id])))


class DirectedGraphConvolution(TrainableOperation):
    id = 0

    def __init__(self, shape: Shape):
        super().__init__()
        DirectedGraphConvolution.id += 1
        self.label = "dgcn_" + str(DirectedGraphConvolution.id)

        # self.input = input
        # self.output = None
        # self.shape = shape
        #
        # self.graph = graph

        self.input = None
        self.output = None
        self.shape = shape

        self.graph = None

        # self.parameters = {}
        # self.parameters[self.label] = {}
        # self.parameters[self.label]["W1"] = tf.Variable(
        #     UtilityOperation.glorot_intializer(self.shape.input, self.shape.output))
        # self.parameters[self.label]["b1"] = tf.Variable(
        #     UtilityOperation.glorot_intializer(self.shape.batch_size, self.shape.output))
        # self.parameters[self.label]["W2"] = tf.Variable(
        #     UtilityOperation.glorot_intializer(self.shape.input, self.shape.output))
        # self.parameters[self.label]["b2"] = tf.Variable(
        #     UtilityOperation.glorot_intializer(self.shape.batch_size, self.shape.output))

        self.parameters = [Parameter(Shape(self.shape.input, self.shape.output, self.shape.batch_size)),
                           Parameter(Shape(self.shape.batch_size, self.shape.output, self.shape.batch_size)),
                           Parameter(Shape(self.shape.input, self.shape.output, self.shape.batch_size)),
                           Parameter(Shape(self.shape.batch_size, self.shape.output, self.shape.batch_size))]

    # def get_parameters(self):
    #     return self.parameters
    #
    # def set_parameters(self, updated_parameters):
    #     self.parameters[self.label]["W1"] = updated_parameters[self.label]["W1"]
    #     self.parameters[self.label]["b1"] = updated_parameters[self.label]["b1"]
    #     self.parameters[self.label]["W2"] = updated_parameters[self.label]["W2"]
    #     self.parameters[self.label]["b2"] = updated_parameters[self.label]["b2"]

    def compute(self):
        a = tf.add(tf.matmul(
            tf.matmul(tf.convert_to_tensor(self.graph.degree_normalized_weighted_out_adjacency, dtype=tf.float32),
                      self.input.output
                      ), parameter_bus[self.parameters[3].id]), parameter_bus[self.parameters[4].id])
        aT = tf.transpose(a)
        b = tf.add(tf.matmul(
            tf.matmul(tf.convert_to_tensor(self.graph.degree_normalized_weighted_in_adjacency, dtype=tf.float32),
                      self.input.output
                      ), parameter_bus[self.parameters[1].id]), parameter_bus[self.parameters[2].id])
        bT = tf.transpose(b)
        self.output = tf.add_n([tf.matmul(a, aT), tf.matmul(a, bT), tf.matmul(b, aT), tf.matmul(b, bT)])


class Dense(TrainableOperation):
    id = 0

    def __init__(self, shape: Shape):
        super().__init__()
        Dense.id += 1
        self.label = "dense_" + str(Dense.id)

        # self.input = input
        # self.output = None
        # self.shape = shape

        self.input = None
        self.output = None
        self.shape = shape

        # self.parameters = {}
        # self.parameters[self.label] = {}
        # self.parameters[self.label]["W1"] = tf.Variable(
        #     UtilityOperation.glorot_intializer(self.shape.input, self.shape.output)).numpy().tolist()
        # self.parameters[self.label]["b1"] = tf.Variable(
        #     UtilityOperation.glorot_intializer(self.shape.batch_size, self.shape.output)).numpy().tolist()

        self.parameters = [Parameter(Shape(self.shape.input, self.shape.output, self.shape.batch_size)),
                           Parameter(Shape(self.shape.batch_size, self.shape.output, self.shape.batch_size))]

    # def get_parameters(self):
    #     return self.parameters
    #
    # def set_parameters(self, updated_parameters):
    #     self.parameters[self.label]["W1"] = updated_parameters[self.label]["W1"]
    #     self.parameters[self.label]["b1"] = updated_parameters[self.label]["b1"]

    def compute(self):
        self.output = tf.add(
            tf.matmul(self.input.output, tf.Variable(tf.convert_to_tensor(parameter_bus[self.parameters[0].id]))),
            tf.Variable(tf.convert_to_tensor(parameter_bus[self.parameters[1].id])))
