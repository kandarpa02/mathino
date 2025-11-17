class Tape:
    def __init__(self):
        self.nodes = []

    def add(self, node):
        self.nodes.append(node)

    def clear(self):
        self.nodes.clear()

tape = Tape()