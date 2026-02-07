# Author: Islam Akef Ebeid


from importer import *
from grapher import *
from graph_encoder import *
from mapper import *


class Loader:
    def __init__(self, data_filename):
        self.data_filename = data_filename
        self.data_df = pd.read_csv(self.data_filename, sep='\t', header=None)

        self.token_to_character_chain = {}
        self.character_chain = [29, 30]
        self.token_chain = [29, 30]
        self.tag_chain = [29]
        self.character_to_token = {}
        self.token_to_tag = {}
        self.character_to_tag = {}
        # self.character_to_tag[""]=[0]
        self.character_clusters = {}
        self.character_to_token_label = {}
        self.character_to_tag_label = {}
        self.token_to_tag_label = {}
        self.character_cluster = []

    @staticmethod
    def normalize_token(input_token):
        output_token = input_token.strip()
        output_token = output_token.lower()
        return output_token

    @staticmethod
    def encode_character(input_character):
        return ord(input_character)

    @staticmethod
    def encode_token(input_token):
        final_encoding = ""
        for c in input_token:
            ascii_code = Loader.encode_character(c)
            final_encoding = final_encoding + str(ascii_code)
        return int(final_encoding)

    @staticmethod
    def convert_chain_to_directed_graph(chain,input_nodes):
        character_edges = []
        for i in tqdm(range(len(chain))):
            if i < len(chain) - 1:
                current_character = chain[i]
                next_character = chain[i + 1]
            else:
                current_character = chain[i]
                next_character = 31
            character_edges.append(str(current_character) + "|" + str(next_character))
        unique_edges = dict(Counter(character_edges))
        filtered_edges = []
        for key, value in unique_edges.items():
            nodes = key.split("|")
            filtered_edges.append((str(nodes[0]), str(nodes[1]), int(value)))
        directed_graph = DirectedGraph(edges=filtered_edges, nodes=input_nodes)
        return directed_graph


    def generate_character_chain(self):
        for key, value in tqdm(self.data_df.iterrows()):
            tag = value[1]
            token = self.normalize_token(value[0])
            characters = [self.encode_character(c) for c in list(token)]
            if characters == [46] and tag == "O":
                characters = characters + [31] + [30]
            else:
                characters = characters + [32]
            self.character_chain = self.character_chain + characters
            self.token_to_character_chain[token] = characters
        del self.character_chain[0]
        del self.character_chain[-1]
        del self.token_to_character_chain[list(self.token_to_character_chain.keys())[-1]][-1]
        del self.token_to_character_chain[list(self.token_to_character_chain.keys())[-1]][0]


    def generate_token_chain(self):
        for key, value in tqdm(self.data_df.iterrows()):
            tag = value[1]
            token = self.normalize_token(value[0])
            token = self.encode_token(token)
            self.token_chain.append(token)
            if token == 46 and tag == "O":
                self.token_chain.append(31)
                self.token_chain.append(30)
        del self.token_chain[-1]

    def generate_tag_chain(self):
        for key, value in tqdm(self.data_df.iterrows()):
            tag = self.normalize_token(value[1])
            tag = self.encode_token(tag)
            self.tag_chain.append(tag)
        self.tag_chain.append(29)

    def create_character_token_map(self):
        for key, value in tqdm(self.data_df.iterrows()):
            token = self.normalize_token(value[0])
            characters = [self.encode_character(c) for c in list(token)]
            token = self.encode_token(token)
            for c in characters:
                if c in self.character_to_token and token not in self.character_to_token[c]:
                    self.character_to_token[c] = self.character_to_token[c] + [token]
                else:
                    self.character_to_token[c] = [token]
        for key, value in self.character_to_token.items():
            if len(value) > 1:
                character_token_graph_label = self.convert_chain_to_directed_graph(chain=value,input_nodes={})
            else:
                character_token_graph_label = value
            self.character_to_token_label[key] = character_token_graph_label

    def create_token_tag_map(self):
        for key, value in tqdm(self.data_df.iterrows()):
            tag = self.normalize_token(value[1])
            token = self.normalize_token(value[0])
            token = self.encode_token(token)
            if token in self.token_to_tag and tag not in self.token_to_tag[token]:
                self.token_to_tag[token] = self.token_to_tag[token] + [tag]
            else:
                self.token_to_tag[token] = [tag]
        for key, value in self.token_to_tag.items():
            if len(value) > 1:
                token_tag_graph_label = self.convert_chain_to_directed_graph(chain=value,input_nodes={})
            else:
                token_tag_graph_label = value
            self.token_to_tag_label[key] = token_tag_graph_label

    # def create_character_tag_map(self):
    #     for key, value in tqdm(self.data_df.iterrows()):
    #         token = self.normalize_token(value[0])
    #         characters = [self.encode_character(c) for c in list(token)]
    #         token = self.encode_token(token)
    #         for c in characters:
    #             if c in self.character_to_tag and token not in self.character_to_tag[c]:
    #                 self.character_to_tag[c] = self.character_to_tag[c] + self.token_to_tag[token]
    #             else:
    #                 self.character_to_tag[c] = self.token_to_tag[token]
    #     for key, value in self.character_to_tag.items():
    #         if len(value) > 1:
    #             character_tag_graph_label = self.convert_chain_to_directed_graph(chain=value,input_nodes={})
    #         else:
    #             character_tag_graph_label = value
    #         self.character_to_tag_label[key] = character_tag_graph_label

    def create_character_tag_map(self):
        for key, value in tqdm(self.token_to_character_chain.items()):
            token = key
            characters = value
            encoded_token = Loader.encode_token(token)
            for c in characters:
                if c in self.character_to_tag and token not in self.character_to_tag[c]:
                    self.character_to_tag[c] = self.character_to_tag[c] + self.token_to_tag[encoded_token]
                else:
                    self.character_to_tag[c] = self.token_to_tag[encoded_token]
        for key, value in self.character_to_tag.items():
            if len(value) > 1:
                character_tag_graph_label = self.convert_chain_to_directed_graph(chain=value,input_nodes={})
            else:
                character_tag_graph_label = value
            self.character_to_tag_label[key] = character_tag_graph_label
        pass


    def cluster_characters(self):
        for char_1, label_graph_1 in self.character_to_tag_label.items():
            if isinstance(label_graph_1, DirectedGraph):
                G1 = label_graph_1.to_networkx()
                node_set1 = set(label_graph_1.nodes.keys())
            else:
                node_set1 = set(label_graph_1)
            for char_2, label_graph_2 in self.character_to_tag_label.items():
                if char_1 == char_2:
                    continue
                if isinstance(label_graph_2, DirectedGraph) and isinstance(label_graph_1, DirectedGraph):
                    G2 = label_graph_2.to_networkx()
                    node_set2 = set(label_graph_2.nodes.keys())
                    check1 = nx.is_isomorphic(G1, G2)
                    # check2 = UtilityOperation.jaccard(list(node_set1), list(node_set2))
                    if check1:
                        self.character_cluster.append(str(char_2))
                elif isinstance(label_graph_2, list) and isinstance(label_graph_1, list):
                    node_set2 = set(label_graph_2)
                    check = UtilityOperation.object_equality(node_set1, node_set2)
                    # check = UtilityOperation.jaccard(list(node_set1), list(node_set2))
                    if check:
                        self.character_cluster.append(str(char_2))
            self.character_clusters[char_1] = "".join(self.character_cluster)
        character_clusters_reversed = UtilityOperation.reverse_dict(self.character_clusters)
        clusters = list(character_clusters_reversed.values())
        return clusters
