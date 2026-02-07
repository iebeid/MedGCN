from importer import *
from grapher import *
from graph_encoder_2 import *


class Model1(Model):
    def __init__(self, data, hyperparameters: Hyperparameters):

        # Assignments
        self.graph = data
        self.hyperparameters = hyperparameters
        self.parameters = []
        self.model = []
        self.order = ["ModelPointer", "Model", "Module", "Layer", "Operation", "Parameter"]

        # Hyperparameters
        self.K = self.hyperparameters["K"]
        self.epochs = self.hyperparameters["epochs"]
        self.patience = self.hyperparameters["patience"]
        self.split = self.hyperparameters["split"]
        self.residual = self.hyperparameters["residual"]
        self.batch_normalization = self.hyperparameters["batch_normalization"]
        self.batch_size = self.hyperparameters["batch_size"]
        self.batch_normalization_factor = self.hyperparameters["batch_normalization_factor"]
        self.regularization_rate = self.hyperparameters["regularization_rate"]
        self.learn_rate = self.hyperparameters["learn_rate"]
        self.dropout_rate = self.hyperparameters["dropout_rate"]
        self.cosine_thershold = self.hyperparameters["cosine_thershold"]
        self.activation = self.hyperparameters["activation"]
        self.input_layer_dimensions = self.hyperparameters["input_layer_dimensions"]
        self.layer_1_dimensions = self.hyperparameters["layer_1_dimensions"]
        self.layer_2_dimensions = self.hyperparameters["layer_2_dimensions"]
        self.layer_3_dimensions = self.hyperparameters["layer_3_dimensions"]
        self.output_layer_dimensions = self.hyperparameters["output_layer_dimensions"]

        # Initialization
        self.root = Node(ModelPointer(Input(shape=self.input_layer_dimensions), self.graph), "root")
        super().__init__(self.root)
        self.insert(self.root, Node(self, "grandparent"))
        self.train_mask, self.test_mask, self.valid_mask = self.graph.balanced_node_label_sampler(self.split)
        self.graph.prepare_training_data(self.input_layer_dimensions.output)

    def compile(self):

        self.model.append(self.root.children[0].data)

        # Input Module
        self.model.append(Module())
        # Input Layer
        self.model.append(Layer(input=Input(shape=self.input_layer_dimensions)))

        # Module 1
        self.model.append(Module())
        # First GCN Layer
        self.model.append(Layer(input=Input(shape=self.layer_1_dimensions)))
        # Graph convolution operation
        self.model.append(GraphConvolution(shape=self.layer_1_dimensions))
        # Batch normalization operation totally optional
        self.model.append(BatchNormalization(bn_factor=self.batch_normalization_factor, shape=self.layer_1_dimensions))
        # Tanh activation operation mandatory but could be replaced with RELU
        self.model.append(TanhActivation(shape=self.layer_1_dimensions))
        # Residual operation is important to facilitate training
        self.model.append(Residual(shape=self.layer_1_dimensions))
        # Regularzation operation is mandatory for training
        self.model.append(Regularizer(rate=self.regularization_rate, shape=self.layer_1_dimensions))
        # Dropout operation is mandatory for training
        self.model.append(Dropout(rate=self.dropout_rate, shape=self.layer_1_dimensions))

        # Second GCN Layer
        self.model.append(Layer(input=Input(shape=self.layer_2_dimensions)))
        # Graph convolution operation
        self.model.append(GraphConvolution(shape=self.layer_2_dimensions))
        # Batch normalization operation
        self.model.append(BatchNormalization(bn_factor=self.batch_normalization_factor, shape=self.layer_2_dimensions))
        # Tanh activation operation
        self.model.append(TanhActivation(shape=self.layer_2_dimensions))
        # Residual operation
        self.model.append(Residual(shape=self.layer_2_dimensions))
        # Dropout operation
        self.model.append(Dropout(rate=self.dropout_rate, shape=self.layer_2_dimensions))

        # Third GCN Layer
        self.model.append(Layer(input=Input(shape=self.layer_3_dimensions)))
        # Graph convolution operation
        self.model.append(GraphConvolution(shape=self.layer_3_dimensions))
        # Batch normalization operation
        self.model.append(BatchNormalization(bn_factor=self.batch_normalization_factor, shape=self.layer_3_dimensions))
        # Tanh activation operation
        self.model.append(TanhActivation(shape=self.layer_3_dimensions))
        # Residual operation
        self.model.append(Residual(shape=self.layer_3_dimensions))
        # Normalization operation
        self.model.append(Normalizer(shape=self.layer_3_dimensions))
        # Dropout operation
        self.model.append(Dropout(rate=self.dropout_rate, shape=self.layer_3_dimensions))

        # Output Module
        self.model.append(Module())
        # Forth dense layer
        self.model.append(Layer(input=Input(shape=self.output_layer_dimensions)))
        self.model.append(Dense(shape=self.output_layer_dimensions))
        # Fifth dense layer
        self.model.append(Dense(shape=self.output_layer_dimensions))
        # Output prediction layer
        self.model.append(SoftmaxPrediction(shape=self.output_layer_dimensions))


    def build(self):
        # Insert Parameter objects into model definition
        model_params = self.model.copy()
        for i, obj in enumerate(self.model):
            lengths_difference = len(model_params) - len(self.model)
            if isinstance(obj, Operation):
                for param in obj.parameters:
                    model_params.insert(i+lengths_difference+1, param)
        self.model = model_params

        # model_as_nodes = []
        current_node = self.model[0]
        next_node = self.model[0]
        number_of_nodes = len(self.model)
        # tree_population_condition = ((isinstance(current_node, Model) and isinstance(next_node, Module))
        #                              or (isinstance(current_node, Module) and isinstance(next_node, Layer))
        #                              or (isinstance(current_node, Layer) and isinstance(next_node, Operation))
        #                              or (isinstance(current_node, Parameter) and isinstance(next_node, Operation))
        #                              or (isinstance(current_node, Operation) and isinstance(next_node, Parameter))
        #                              or (isinstance(current_node, Parameter) and isinstance(next_node, Parameter))
        #                              or (isinstance(current_node, Operation) and isinstance(next_node, Operation)))
        for i in range(number_of_nodes):
            if i < (number_of_nodes-1):
                current_node = self.model[i]
                next_node = self.model[i+1]
            else:
                current_node = self.model[i]
                next_node = None

            pass

            if isinstance(current_node, Model) and isinstance(next_node, Module):
                self.insert(Node(current_node, "parent1"), Node(next_node, "parent1"))
            if isinstance(current_node, Module) and isinstance(next_node, Layer):
                self.insert(Node(current_node, "parent2"), Node(next_node, "parent2"))

            if isinstance(current_node, Layer) and isinstance(next_node, Module):
                self.insert(Node(current_node, "parent3"), Node(next_node, "parent3"))

            if isinstance(current_node, Layer) and isinstance(next_node, Operation):
                self.insert(Node(current_node, "parent3"), Node(next_node, "parent3"))
            if isinstance(current_node, Parameter) and isinstance(next_node, Operation):
                self.insert(Node(current_node, "parent3"), Node(next_node, "parent3"))
            if isinstance(current_node, Operation) and isinstance(next_node, Parameter):
                self.insert(Node(current_node, "leaf"), Node(next_node, "leaf"))
            if isinstance(current_node, Parameter) and isinstance(next_node, Parameter):
                self.insert(Node(current_node, "leaf"), Node(next_node, "leaf"))
            if isinstance(current_node, Operation) and isinstance(next_node, Operation):
                self.insert(Node(current_node, "leaf"), Node(next_node, "leaf"))

            if isinstance(current_node, Operation) and next_node is None:
                current_node.status = "leaf"
            if isinstance(current_node, Parameter) and next_node is None:
                current_node.status = "leaf"

            pass




        # for obj in self.model:
        #     previous_parent = current_parent
        #     if isinstance(obj, Module):
        #         # self.print_tree(node=self.root)
        #         # print("--------------------")
        #         # model_as_nodes.append(Node(obj, "parent1"))
        #         parent = self.insert(current_parent, Node(obj, "parent1"))
        #         # parent_tree = Tree(parent)
        #         # self.print_tree(node=self.root)
        #         # pass
        #         # print("--------------------")
        #         # parent_tree.print_tree(node=parent_tree.root)
        #         # print("--------------------")
        #         # self.print_tree(node=self.root)
        #         # print("--------------------")
        #         # self.merge(current_parent.data, parent_tree)
        #         # self.print_tree(node=self.root)
        #         # print("--------------------")
        #         # pass
        #
        #         # source_tree = Tree(object7)
        #         # source_tree.insert(object7, object8)
        #         # source_tree.insert(object7, object9)
        #         #
        #         # # Merge subtree of source_tree with node X
        #         # tree.merge(object6, source_tree)
        #         # print("Tree after merge:")
        #         # tree.print_tree(node=tree.root)
        #
        #     if isinstance(obj, Layer):
        #         # model_as_nodes.append(Node(obj, "parent2"))
        #         parent = self.insert(current_parent, Node(obj, "parent2"))
        #     if isinstance(obj, Operation):
        #         # model_as_nodes.append(Node(obj, "parent3"))
        #         parent = self.insert(current_parent, Node(obj, "parent3"))
        #     if isinstance(obj, Parameter):
        #         # model_as_nodes.append(Node(obj, "leaf"))
        #         parent = self.insert(current_parent, Node(obj, "leaf"))
        #
        #     current_parent = previous_parent
        #
        # # self.root = current_parent
        # self.print_tree(node=self.root)
        # pass


        # # Insert model definition in model tree
        # parent_obj = None
        # child_obj = None
        # size_of_definition = len(self.model)
        # for i in range(size_of_definition):
        #
        #     if i < size_of_definition:
        #         parent_obj = self.model[i]
        #         child_obj = self.model[i+1]
        #     else:
        #         parent_obj = self.model[i]
        #         child_obj = None
        #
        #     pass
        #
        #     if type(parent_obj) != type(child_obj):
        #         parent_obj.children.append(Node(child_obj))
        #     else:
        #         parent_of_previous = self._find_node_with_parent(parent_obj)
        #         parent_of_previous.children.append(Node(child_obj))
        #
        #     pass













        # # Insert model definition object into model tree
        # current_obj = None
        # previous_obj = self.root.children[0].data
        # for obj in self.model:
        #     current_obj = Node(obj)
        #     # if :
        #         # self.root = self.insert(self.root.children , Node(current_obj, "parent"))
        #     # if :
        #     #     previous_obj = self.find(previous_obj)
        #     #     previous_obj = self.insert(previous_obj, Node(current_obj, "parent"))
        #     if (isinstance(current_obj, Module)) or (isinstance(obj, Layer) and isinstance(previous_obj, Module)) or (isinstance(obj, Operation) and isinstance(previous_obj, Layer)):
        #         # previous_obj = self.find(previous_obj)
        #         current_obj.status = "parent"
        #         previous_obj = self.insert(previous_obj, current_obj)
        #     if isinstance(obj, Parameter)and isinstance(previous_obj, Operation):
        #         # previous_obj = self.find(previous_obj)
        #         current_obj.status = "leaf"
        #         previous_obj = self.insert(previous_obj, current_obj)
        #     previous_obj = current_obj
        # self.root = previous_obj


    # def build(self):
    #     # Build function:
    #     # 1- Inserts all objects in model tree
    #     # 2- Executes all objects in model definition
    #     # 4- Executes the compiled objects
    #     # 5- Initializes and populates the parameter bus from parameter objects

    #     temp_dict = {}
    #     for i, obj in enumerate(self.model):
    #         temp_dict[i] = obj
    #     # inverted_temp_dict = UtilityOperation.invert_dict(temp_dict)
    #     inverted_temp_dict = UtilityOperation.invert_dict_by_object_type(temp_dict)
    #
    #     # unique_ordered_objects = np.unique(self.model)
    #     # model_objects = []
    #     # module_objects = []
    #     # layer_objects = []
    #     # operation_objects = []
    #     # parameter_objects = []
    #
    #     # for key, value in self.model.items():
    #     #     if isinstance(value, Model):
    #     #         # inverted_dict.setdefault(value, list()).append(key)
    #     #         model_objects.append()
    #     #     if isinstance(value, Module):
    #     #         # inverted_dict.setdefault(value, list()).append(key)
    #     #
    #     #     if isinstance(value, Layer):
    #     #         # inverted_dict.setdefault(value, list()).append(key)
    #     #
    #     #     if isinstance(value, Operation):
    #     #         # inverted_dict.setdefault(value, list()).append(key)
    #     #
    #     #     if isinstance(value, Parameter):
    #     #         # inverted_dict.setdefault(value, list()).append(key)
    #
    #     def get_type_of_item(obj):
    #         if isinstance(obj, ModelPointer):
    #             return "ModelPointer"
    #         if isinstance(obj, Model):
    #             return "Model"
    #         if isinstance(obj, Module):
    #             return "Module"
    #         if isinstance(obj, Layer):
    #             return "Layer"
    #         if isinstance(obj, Operation):
    #             return "Operation"
    #         if isinstance(obj, Parameter):
    #             return "Parameter"
    #
    #     def object_type_history(item):
    #         for i in range(len(self.model)):
    #             current_item = self.model[i]
    #             current_item_type = get_type_of_item(current_item)
    #             current_item_type_index_order = self.order.index(current_item_type)
    #             if current_item == item:
    #                 for k in range(len(self.model[0:i])):
    #                     k = k + 1
    #                     previous_item = self.model[i - k]
    #                     previous_item_type = get_type_of_item(previous_item)
    #                     previous_item_type_index_order = self.order.index(previous_item_type)
    #                     if (previous_item_type in self.order) and (current_item_type in self.order):
    #                         if (current_item_type_index_order - previous_item_type_index_order == 1):
    #                             pass
    #
    #     current_obj = None
    #     previous_obj = self.root
    #     for obj in self.model:
    #         current_obj = obj
    #         type = get_type_of_item(current_obj)
    #         if isinstance(obj, Module) or isinstance(obj, Layer):
    #             previous_obj = self.insert(previous_obj, current_obj)
    #         if isinstance(obj, Operation) or isinstance(obj, Parameter):
    #             pass  # find the most recent previous object type
    #
    #         previous_obj = current_obj
    #
    #     # current_obj = None
    #     # previous_obj = None
    #     # for k in unique_ordered_objects:
    #     #     current_obj = k
    #     #     if isinstance(obj, ModelPointer):
    #     #         continue
    #     #     else:
    #     #         if isinstance(obj, Model):
    #     #             self.root = self.insert(self.root, obj)
    #     #         elif isinstance(previous_obj, Operation):
    #     #             previous_obj = self.insert(previous_obj, current_obj)
    #     #             for param in previous_obj.parameters:
    #     #                 previous_obj = self.insert(previous_obj, param)
    #     #         else:
    #     #             previous_obj = self.insert(previous_obj, current_obj)
    #     #     previous_obj = current_obj
    #
    # # def init_parameters(self):
    # #     pass
    # #
    # # def set_parameters(self, updated_parameters):
    # #     pass
    # #
    # # def get_parameters(self):
    # #     self.parameters = []
    # #     self.parameters = self.parameters + self.input.parameters
    # #     self.parameters = self.parameters + self.H1.parameters
    # #     self.parameters = self.parameters + self.H2.parameters
    # #     self.parameters = self.parameters + self.H3.parameters
    # #     self.parameters = self.parameters + self.H4.parameters
    # #     self.parameters = self.parameters + self.output.parameters

    def compute(self):
        # self.input.compute()
        # self.H1.compute()
        # self.regularization_constant = self.H1.regularization_constant
        # self.H2.compute()
        # self.H3.compute()
        # self.embeddings = self.H3.embedding
        # self.H4.compute()
        # self.output.compute()
        # self.predictions = self.output.output
        pass

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
            params = self.get_parameters()
            self.gradients = tape.gradient(self.train_loss, params)
            UtilityOperation.optimizer(self.learn_rate).apply_gradients(list(zip(self.gradients, params)))
            self.set_parameters(params)

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


if __name__ == "__main__":
    # Prep Data
    G = nx.karate_club_graph()
    raw_data = []
    for edge in G.edges():
        raw_data.append((str(edge[0]), str(edge[1]), float(random.randint(1, 10))))
    graph = UndirectedGraph(edges=raw_data)
    graph.create_node_labels(True)
    print(graph)
    # Prep Hyperparameters
    hp = Hyperparameters()
    hp.add("K", 10)
    hp.add("epochs", 40)
    hp.add("patience", 100)
    hp.add("split", 80)
    hp.add("residual", True)
    hp.add("batch_normalization", True)
    hp.add("batch_size", graph.number_of_nodes)
    hp.add("batch_normalization_factor", 0.001)
    hp.add("regularization_rate", 0.0005)
    hp.add("learn_rate", 0.01)
    hp.add("dropout_rate", 0.5)
    hp.add("cosine_thershold", 0.001)
    hp.add("activation", "tanh")
    hp.add("input_layer_dimensions",
           Shape(hp["batch_size"], 16, hp["batch_size"]))
    hp.add("layer_1_dimensions", Shape(16, 16, hp["batch_size"]))
    hp.add("layer_2_dimensions", Shape(16, 16, hp["batch_size"]))
    hp.add("layer_3_dimensions", Shape(16, 16, hp["batch_size"]))
    hp.add("output_layer_dimensions",
           Shape(16, graph.number_of_classes, hp["batch_size"]))
    # Define Model
    m1 = Model1(graph, hp)
    m1.compile()
    m1.build()
    # m1.compute()
    # m1.train()
    # m1.validate()
    # m1.visualize()
    m1.print_tree()
