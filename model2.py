from importer import *
from grapher import *
from graph_encoder import *

class Model2(Model):
    def __init__(self, data, hyperparameters: Attribute):
        super().__init__()
        # Data
        self.graph = data

        # Hyperparameters
        self.epochs = hyperparameters.configuration["epochs"]
        self.split = hyperparameters.configuration["split"]
        self.batch_normalization_factor = hyperparameters.configuration["batch_normalization_factor"]
        self.regularization_rate = hyperparameters.configuration["regularization_rate"]
        self.dropout_rate = hyperparameters.configuration["dropout_rate"]
        self.learn_rate = hyperparameters.configuration["learn_rate"]
        self.activation = hyperparameters.configuration["activation"]
        self.input_layer_dimensions = hyperparameters.configuration["input_layer_dimensions"]
        self.layer_1_dimensions = hyperparameters.configuration["layer_1_dimensions"]
        self.layer_2_dimensions = hyperparameters.configuration["layer_2_dimensions"]
        self.layer_3_dimensions = hyperparameters.configuration["layer_3_dimensions"]
        self.output_layer_dimensions = hyperparameters.configuration["output_layer_dimensions"]


        # Extract training, test and validation data
        self.train_mask, self.test_mask, self.valid_mask = self.graph.balanced_node_label_sampler(self.split)
        self.graph.prepare_training_data(self.input_layer_dimensions.output)


    def compile(self):

        # Input Layer
        self.input = Layer(id=1,input=Input(id=1,shape=self.input_layer_dimensions))
        self.add(self.input)

        # First GCN Layer
        self.H1 = Layer(id=2,input=Input(input=self.input, shape=self.layer_1_dimensions))
        operation1 = GraphConvolution(input=self.H1.input, graph=self.graph, shape=self.layer_1_dimensions)
        operation3 = TanhActivation(input=operation1, shape=self.layer_1_dimensions)
        operation4 = Residual(input1=operation1, input2=operation3, shape=self.layer_1_dimensions)
        operation5 = Regularizer(input=operation4, rate=self.regularization_rate)
        operation6 = Dropout(input=operation5, rate=self.dropout_rate)
        self.H1.add(operation1)
        self.H1.add(operation3)
        self.H1.add(operation4)
        self.H1.add(operation5)
        self.H1.add(operation6)
        self.add(self.H1)

        # Second GCN Layer
        self.H2 = Layer(input=Input(input=self.H1, shape=self.layer_2_dimensions))
        operation1 = GraphConvolution(input=self.H2.input, graph=self.graph, shape=self.layer_2_dimensions)
        operation3 = TanhActivation(input=operation1, shape=self.layer_2_dimensions)
        operation4 = Residual(input1=operation1, input2=operation3, shape=self.layer_2_dimensions)
        operation5 = Dropout(input=operation4, rate=self.dropout_rate)
        self.H2.add(operation1)
        self.H2.add(operation3)
        self.H2.add(operation4)
        self.H2.add(operation5)
        self.add(self.H2)

        # Third GCN Layer
        self.H3 = Layer(input=Input(input=self.H2, shape=self.layer_3_dimensions))
        operation1 = GraphConvolution(input=self.H3.input, graph=self.graph, shape=self.layer_3_dimensions)
        operation3 = TanhActivation(input=operation1, shape=self.layer_3_dimensions)
        operation4 = Residual(input1=operation1, input2=operation3, shape=self.layer_3_dimensions)
        operation5 = Normalizer(input=operation4)
        operation6 = Dropout(input=operation5, rate=self.dropout_rate)
        self.H3.add(operation1)
        self.H3.add(operation3)
        self.H3.add(operation4)
        self.H3.add(operation5)
        self.H3.add(operation6)
        self.add(self.H3)

        # Forth dense layer
        self.H4 = Layer(input=Input(input=self.H3, shape=self.output_layer_dimensions))
        operation1 = Dense(input=self.H4.output, shape=self.output_layer_dimensions)
        self.H4.add(operation1)
        self.add(self.H4)

        # Output prediction layer
        self.output = Layer(input=Input(input=self.H4, shape=self.output_layer_dimensions))
        operation1 = SoftmaxPrediction(input=self.output.output, shape=self.output_layer_dimensions)
        self.output.add(operation1)
        self.add(self.output)

    def compute(self):
        self.input.compute()
        self.H1.compute()
        self.regularization_constant = self.H1.regularization_constant
        self.H2.compute()
        self.H3.compute()
        self.embeddings = self.H3.embedding
        self.H4.compute()
        self.output.compute()
        self.predictions = self.output.output


    def collect(self):
        self.parameters = []
        self.parameter_lookup = []
        self.parameters = self.parameters + self.input.parameters
        self.parameter_lookup.append(self.input.parameters)
        self.parameters = self.parameters + self.H1.parameters
        self.parameter_lookup.append(self.H1.parameters)
        self.parameters = self.parameters + self.H2.parameters
        self.parameter_lookup.append(self.H2.parameters)
        self.parameters = self.parameters + self.H3.parameters
        self.parameter_lookup.append(self.H3.parameters)
        self.parameters = self.parameters + self.H4.parameters
        self.parameter_lookup.append(self.H4.parameters)
        self.parameters = self.parameters + self.output.parameters
        self.parameter_lookup.append(self.output.parameters)


    def update(self):
        # self.input.update(self.parameter_lookup[0])
        # self.H1.update(self.parameter_lookup[1])
        # self.H2.update(self.parameter_lookup[2])
        # self.H3.update(self.parameter_lookup[3])
        # self.H4.update(self.parameter_lookup[4])
        # self.output.update(self.parameter_lookup[5])

        self.input.update(self.parameters)
        self.H1.update(self.parameters)
        self.H2.update(self.parameters)
        self.H3.update(self.parameters)
        self.H4.update(self.parameters)
        self.output.update(self.parameters)

    # @tf.function
    def train(self):
        for epoch in tf.range(self.epochs):
            start_time = time.perf_counter()
            # Train
            with tf.GradientTape() as tape:
                self.compute()
                self.train_loss = UtilityOperation.masked_cross_entropy_loss_evaluater(self.predictions, self.graph.y,
                                                                                       self.regularization_constant,
                                                                                       self.train_mask)
                self.train_accuracy = UtilityOperation.masked_accuracy_evaluater(self.predictions, self.graph.y,
                                                                                 self.train_mask)
            self.collect()
            self.gradients = tape.gradient(self.train_loss, self.parameters)
            UtilityOperation.optimizer(self.learn_rate).apply_gradients(list(zip(self.gradients, self.parameters)))
            self.update()

            # Test
            self.compute()
            self.test_loss = UtilityOperation.masked_cross_entropy_loss_evaluater(self.predictions, self.graph.y,
                                                                                  self.regularization_constant,
                                                                                  self.test_mask)
            self.test_accuracy = UtilityOperation.masked_accuracy_evaluater(self.predictions, self.graph.y,
                                                                            self.test_mask)
            end_time = time.perf_counter()

            # Print Information
            self.time_per_epoch = tf.constant(round((end_time - start_time), 3), dtype=tf.float32)
            tf.print(" Epoch: " + tf.strings.as_string(epoch)
                     + " Seconds/Epoch: " + tf.strings.as_string(self.time_per_epoch)
                     + " Learning Rate: " + tf.strings.as_string(
                tf.constant(round(self.learn_rate, 3), dtype=tf.float32))
                     + " Train Loss: " + tf.strings.as_string(self.train_loss)
                     + " Train Accuracy: " + tf.strings.as_string(self.train_accuracy)
                     + " Test Loss: " + tf.strings.as_string(self.test_loss)
                     + " Test Accuracy: " + tf.strings.as_string(self.test_accuracy)
                     )

    def validate(self):
        # Validate
        self.compute()
        self.valid_loss = UtilityOperation.masked_cross_entropy_loss_evaluater(self.predictions, self.graph.y,
                                                                               self.regularization_constant,
                                                                               self.valid_mask)
        self.valid_accuracy = UtilityOperation.masked_accuracy_evaluater(self.predictions, self.graph.y,
                                                                         self.valid_mask)




    def visualize(self):
        tsne = skl.manifold.TSNE(n_components=2, perplexity=3, learning_rate=10)
        tsne.fit_transform(np.array(self.embeddings))
        z = tsne.embedding_[:, 0]
        y = tsne.embedding_[:, 1]
        fig, ax = plt.subplots()
        ax.scatter(z, y)
        for i, txt in enumerate(self.graph.nodes.values()):
            ax.annotate(txt, (z[i], y[i]))
        plt.show()
