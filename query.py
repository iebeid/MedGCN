from importer import *

class Query:
    def __init__(self):
        pass

    def embed_entities(self, character_graph, character_embedding):
        self.entity_embeddings = {}
        for doc_id, body in self.documents.items():
            converted_body = []
            for c in body:
                character_index = character_graph.nodes[c]
                converted_body.append(character_index)
            found_embedding = tf.nn.embedding_lookup(tf.convert_to_tensor(character_embedding), converted_body)
            entity_embedding_tensor = tf.math.reduce_mean(found_embedding, axis=0)
            self.entity_embeddings[doc_id] = entity_embedding_tensor.numpy()

    def find_matches(self, threshold):
        pairs = []
        for docid1, dvector1 in self.entity_embedding.items():
            for docid2, dvector2 in self.entity_embedding.items():
                if docid1 <= docid2:
                    continue
                cosine_distance = self.cosine_distance(dvector1, dvector2)
                if cosine_distance > 0 and cosine_distance < threshold:
                    pairs.append((docid1, docid2))
        return pairs