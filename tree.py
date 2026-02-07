class Node:
    def __init__(self, data, status=None):
        self.data = data
        self.status = status
        self.children = []
        self.parent = None


class Tree:
    def __init__(self, root):
        if root is None:
            self.root = Node(self, "root")
        else:
            self.root = root
        self.leaves = []
        self.pointer = self.root

    # Insert a new node as a child of a specific parent node
    def insert(self, parent, child):
        self.pointer = self._find_node(parent.data)
        if self.pointer:
            self.pointer.children.append(Node(child.data, "leaf"))
            self.pointer.status = "parent"
        else:
            print(f"Parent node with data {parent} not found")
        # merge
        self.sub_tree = Tree(self.pointer)
        self.merge(self.pointer.data, self.sub_tree)
        pass



    # Find a node by its data
    def find(self, data):
        return self._find_node(data)

    # Helper function to find a node recursively
    def _find_node(self, data, node=None):
        if node is None:
            node = self.root
        if node.data == data and type(node.data) == type(data):
            return node
        for child in node.children:
            found = self._find_node(data, child)
            if found:
                return found
        return None

    def get_leaves(self):
        self._traverse_tree()

    # Helper function to traverse tree root to leaf and return leaves
    def _traverse_tree(self, node=None):
        if node is None:
            node = self.root
        if node.status == "leaf":
            self.leaves.append(node)
        for child in node.children:
            found = self._traverse_tree(child)
            if found:
                return found
        return None

    # Update the data of a node
    def update(self, data, new_data):
        node = self.find(data)
        if node:
            node.data = new_data
        else:
            print(f"Node with data {data} not found")

    # Delete a node by its data
    def delete(self, data):
        parent, child = self._find_node_with_parent(data)
        if child:
            parent.children.remove(child)
        else:
            print(f"Node with data {data} not found")

    # Helper function to find a node and its parent
    def _find_node_with_parent(self, data, node=None, parent=None):
        if node is None:
            node = self.root
        if node.data == data:
            return parent, node
        for child in node.children:
            found_parent, found_child = self._find_node_with_parent(data, child, node)
            if found_child:
                return found_parent, found_child
        return None, None

    # Merge two trees with a specified target node in the first tree
    def merge(self, target_data, source_tree):
        target_node = self.find(target_data)
        if target_node and source_tree.root:
            target_node.children.extend(source_tree.root.children)
        else:
            print(f"Target node: {target_data} not found or source tree is empty")

    # Print the tree structure (simple pre-order traversal)
    def print_tree(self, node=None, level=0):
        if node is None:
            return
        print("  " * level, node.data)
        for child in node.children:
            self.print_tree(child, level + 1)

class Model(Tree):
    def __init__(self, root):
        super().__init__(root)
        self.root = root


    def compile(self):
        self.m1 = Module("parent")
        self.l11 = Layer("parent")
        self.o11 = Operation("parent")
        self.param11A = Parameter("leaf")
        self.param12A = Parameter("leaf")
        self.o12 = Operation("parent")
        self.param11B = Parameter("leaf")
        self.o13 = Operation("parent")
        self.param11C = Parameter("leaf")
        self.param12C = Parameter("leaf")
        self.l12 = Layer("parent")
        self.o14 = Operation("parent")
        self.param11D = Parameter("leaf")
        self.param12D = Parameter("leaf")
        self.o15 = Operation("parent")
        self.param11E = Parameter("leaf")
        self.param12E = Parameter("leaf")
        self.m2 = Module("parent")
        self.l21 = Layer("parent")
        self.o21 = Operation("parent")
        self.param21A = Parameter("leaf")
        self.param22A = Parameter("leaf")
        self.o22 = Operation("parent")
        self.param21B = Parameter("leaf")
        self.o23 = Operation("parent")
        self.param21C = Parameter("leaf")
        self.param22C = Parameter("leaf")
        self.l22 = Layer("parent")
        self.o24 = Operation("parent")
        self.param21D = Parameter("leaf")
        self.param22D = Parameter("leaf")
        self.o25 = Operation("parent")
        self.param21E = Parameter("leaf")
        self.param22E = Parameter("leaf")



        self.root = self.insert(self.root, self.m1)
        self.root = self.insert(self.root, self.m2)

        # --------------------------------

        self.m1 = self.insert(self.m1, self.l11)
        self.m1 = self.insert(self.m1, self.l12)

        self.l11 = self.insert(self.l11, self.o11)
        self.l11 = self.insert(self.l11, self.o12)
        self.l11 = self.insert(self.l11, self.o13)

        self.l12 = self.insert(self.l12, self.o14)
        self.insert(self.l12, self.o15)

        self.insert(self.o11, self.param11A)
        self.insert(self.o11, self.param12A)

        self.insert(self.o12, self.param11B)

        self.insert(self.o13, self.param11C)
        self.insert(self.o13, self.param12C)

        self.insert(self.o14, self.param11D)
        self.insert(self.o14, self.param12D)

        self.insert(self.o15, self.param11E)
        self.insert(self.o15, self.param12E)

        # -------------------------------

        self.insert(self.m2, self.l21)
        self.insert(self.m2, self.l22)

        self.insert(self.l21, self.o21)
        self.insert(self.l21, self.o22)
        self.insert(self.l21, self.o23)

        self.insert(self.l22, self.o24)
        self.insert(self.l22, self.o25)

        self.insert(self.o21, self.param21A)
        self.insert(self.o21, self.param22A)

        self.insert(self.o22, self.param21B)

        self.insert(self.o23, self.param21C)
        self.insert(self.o23, self.param22C)

        self.insert(self.o24, self.param21D)
        self.insert(self.o24, self.param22D)

        self.insert(self.o25, self.param21E)
        self.insert(self.o25, self.param22E)

        # self.print_tree(model_root)

        # ----------------------------------------


class ModelPointer(Node):
    def __init__(self, status):
        super().__init__(self, status)
        pass


class Module(Node):
    def __init__(self, status):
        super().__init__(self, status)
        pass


class Layer(Node):
    def __init__(self, status):
        super().__init__(self, status)
        pass


class Operation(Node):
    def __init__(self, status):
        super().__init__(self, status)
        pass


class Parameter(Node):
    id = 0

    def __init__(self, status):
        super().__init__(self, status)
        Parameter.id += 1


if __name__ == "__main__":
    # Initialize objects

    model_root = ModelPointer("root")
    model1 = Model(model_root)
    model1.compile()

    # Create another tree for merging

    operation5 = Operation(status="root")
    object7 = Parameter(status="leaf")
    object8 = Parameter(status="leaf")
    object9 = Parameter(status="leaf")

    source_tree = Tree(operation5)
    source_tree.insert(operation5, object7)
    source_tree.insert(operation5, object8)
    source_tree.insert(operation5, object9)
    print("--Tree to be merged--")
    source_tree.print_tree(node=source_tree.root)
    print("---------------------")
    # Merge subtree of source_tree with node X
    print("---------------------")
    print("Tree before merge:")
    model1.print_tree(node=model1.root)
    model1.merge(model1.param22E, source_tree)
    print("---------------------")
    print("Tree after merge:")
    model1.print_tree(node=model1.root)
    print("---------------------")


# m1.insert(m1, object1)
# object2 = Parameter()
# m1.insert(m1, object2)
# object3 = Parameter()
# m1.insert(m1, object3)
# object4 = Parameter()
# m1.insert(m1, object4)
#
# model1.merge(model1.root, m1)
# model1.get_leaves()
#
# # Test Tree Implmentation
# object1 = Parameter()
# object2 = Parameter()
# object3 = Parameter()
# object4 = Parameter()
# object5 = Parameter()
# object6 = Parameter()
# object7 = Parameter()
# object8 = Parameter()
# object9 = Parameter()
#
# # Example usage
# tree = Tree(object1)
# tree.insert(object1, object2)
# tree.insert(object1, object3)
# tree.insert(object2, object4)
# tree.insert(object3, object5)
#
# print("Tree:")
# tree.print_tree(node=tree.root)
#
# # Example usage of find, update, delete, merge
# found_node = tree.find(object3)
# if found_node:
#     print(f"Found node with data: {found_node.data}")
#
# tree.update(object2, object6)
# print(f"Updated node B to X")
#
# tree.delete(object5)
# print(f"Deleted node E")
#
# # Create another tree for merging
# source_tree = Tree(object7)
# source_tree.insert(object7, object8)
# source_tree.insert(object7, object9)
#
# # Merge subtree of source_tree with node X
# tree.merge(object6, source_tree)
# print("Tree after merge:")
# tree.print_tree(node=tree.root)
