from importer import *
from grapher import *
from graph_encoder import *
from mapper import *
from loader import *
from evaluater import *
from model1 import *
from model2 import *
from model3 import *
from model4 import *


class Experiment:
    def __init__(self, dummy=False):

        # Initialize devices
        print("Title: MedGCN, a semantic biomedical named entity disambiguation framework")
        print("Author: Islam Akef Ebeid")
        print("Version: 0.1")
        print("Tensorflow Version: " + str(tf.__version__))
        print("CPU Device: " + str(tf.config.list_physical_devices('CPU')))
        print("GPU Device: " + str(tf.config.list_physical_devices('GPU')))

        # Load data
        if not dummy:
            self.data_filename = "data/devel.tsv"
            loader = Loader(self.data_filename)
            loader.generate_character_chain()
            # loader.generate_token_chain(self.data_df)
            # directed_token_graph = loader.convert_chain_to_directed_graph(self.token_chain)
            # print(directed_token_graph)
            # loader.generate_tag_chain(self.data_df)
            # directed_tag_graph = loader.convert_chain_to_directed_graph(self.tag_chain)
            # print(directed_tag_graph)
            loader.create_token_tag_map()
            loader.create_character_tag_map()
            nodes_clusters = loader.cluster_characters()
            print(nodes_clusters)
            nodes = {}
            cluster_number = 0
            for c in nodes_clusters:
                for n in c:
                    nodes[str(n)] = str(cluster_number)
                cluster_number = cluster_number + 1
            while 29 in loader.character_chain:
                loader.character_chain.remove(29)
            while '29' in loader.character_chain:
                loader.character_chain.remove('29')
            print(loader.character_chain[-1])
            print(loader.character_chain[-2])
            print(loader.character_chain[-3])
            self.graph = loader.convert_chain_to_directed_graph(chain=loader.character_chain, input_nodes=nodes)
            self.graph.create_node_labels(dummy)
            print(self.graph)
        else:
            # Create a karate club graph
            G = nx.karate_club_graph()
            # Add weights to the edges
            self.raw_data = []
            for edge in G.edges():
                self.raw_data.append((str(edge[0]), str(edge[1]), float(random.randint(1, 10))))
            self.graph = UndirectedGraph(edges=self.raw_data)
            self.graph.create_node_labels(dummy)
            print(self.graph)

        # Hyperparameters
        self.K = 10
        self.hp = Hyperparameters()
        self.hp.add("K", self.K)
        self.hp.add("epochs", 40)
        self.hp.add("patience", 100)
        self.hp.add("split", 80)
        self.hp.add("residual", True)
        self.hp.add("batch_normalization", True)
        self.hp.add("batch_size", self.graph.number_of_nodes)
        self.hp.add("batch_normalization_factor", 0.001)
        self.hp.add("regularization_rate", 0.0005)
        self.hp.add("learn_rate", 0.01)
        self.hp.add("dropout_rate", 0.5)
        self.hp.add("cosine_thershold", 0.001)
        self.hp.add("activation", "tanh")
        self.hp.add("input_layer_dimensions",
                    Shape(self.hp["batch_size"], 16, self.hp["batch_size"]))
        self.hp.add("layer_1_dimensions", Shape(16, 16, self.hp["batch_size"]))
        self.hp.add("layer_2_dimensions", Shape(16, 16, self.hp["batch_size"]))
        self.hp.add("layer_3_dimensions", Shape(16, 16, self.hp["batch_size"]))
        self.hp.add("output_layer_dimensions",
                    Shape(16, self.graph.number_of_classes, self.hp["batch_size"]))

        # Metadata
        self.valid_embeddings = []
        self.valid_losses = []
        self.valid_accuracies = []
        self.valid_predictions = []
        self.valid_parameters = []

    def run(self):
        for k in range(self.K):
            print(str(self.K) + " fold validation iteration number: " + str(k))
            m = Model1(self.graph, self.hp) # Vanilla graph convolution with batch normalization
            # m = Model2(self.graph, self.hp) # Vanilla graph convolution without batch normalization
            # m = Model3(self.graph, self.hp)  # Directed graph convolution without batch normalization
            m.compile()
            m.compute()
            m.train()
            m.validate()
            for sublist in m.predictions:
                if np.isnan(sublist[0]):
                    print("The list contains a NaN")
                    break
                else:
                    self.valid_predictions.append(m.predictions)
                    self.valid_losses.append(m.valid_loss)
                    self.valid_accuracies.append(m.valid_accuracy)
                    self.valid_embeddings.append(m.embeddings)
                    self.valid_parameters.append(m.parameters)

    def tester(self):
        m = Model1(self.graph, self.hp)
        m.compile()
        # m.compute()
        m.init_parameters()
        params = m.get_parameters()
        # print(m.parameters)
        print(len(params))
        m.set_parameters(params)

    def finalize(self):
        self.max_valid_accuracy_index = np.argmax(np.array(self.valid_accuracies))
        self.final_predictions = list(self.valid_predictions[self.max_valid_accuracy_index].numpy())
        self.final_loss = self.valid_losses[self.max_valid_accuracy_index].numpy()
        self.final_accuracy = self.valid_accuracies[self.max_valid_accuracy_index].numpy()
        self.final_embeddings = list(self.valid_embeddings[self.max_valid_accuracy_index].numpy())
        self.final_parameters = self.valid_parameters[self.max_valid_accuracy_index]

    def visualize_embeddings(self):
        tsne = skl.manifold.TSNE(n_components=2, perplexity=3, learning_rate=10)
        tsne.fit_transform(np.array(self.final_embeddings))
        z = tsne.embedding_[:, 0]
        y = tsne.embedding_[:, 1]
        fig, ax = plt.subplots()
        ax.scatter(z, y)
        for i, txt in enumerate(self.graph.nodes.values()):
            ax.annotate(txt, (z[i], y[i]))
        plt.show()

    def evaluate(self):
        e = EvaluatER(None, None)
        self.sample_precision, self.sample_recall, self.sample_accuracy, self.sample_f1_score = e.evaluate_embeddings(
            self.graph, self.final_predictions, self.final_embeddings, self.hp.configuration["cosine_thershold"],
            self.hp.configuration["input_layer_dimensions"])
        self.visualize_embeddings()
        self.embedding_precision, self.embedding_recall, self.embedding_accuracy, self.embedding_f1_score = e.evaluate_samples(
            self.graph, self.final_predictions, self.hp.configuration["input_layer_dimensions"])
        self.ari = e.evaluate_clusters(self.graph, self.final_predictions)
        print("Final Loss: " + str(self.final_loss))
        print("Final Total Accuracy: " + str(self.final_accuracy))
        print("Final Samples F1-Score: " + str(self.sample_f1_score))
        print("Final Embedding F1-Score: " + str(self.embedding_f1_score))
        print("Final Adjusted Rand Index: " + str(self.ari))


def measure_execution_time(func):
    def wrapper():
        start_time = time.perf_counter()
        func()
        end_time = time.perf_counter()
        print("Total runtime: " + str(round((end_time - start_time), 4)) + " seconds")

    return wrapper


@measure_execution_time
def main():
    e = Experiment(dummy=True)
    e.run()
    # e.finalize()
    # e.evaluate()
    # e.tester()


if __name__ == "__main__":
    main()
