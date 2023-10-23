import dataclasses
from dataclasses import dataclass
from typing import List
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import DataLoader
import pandas as pd

from .log import *
from sklearn.metrics import confusion_matrix
from sklearn import metrics


class AST:
    def __init__(self, nodes, indentation):
        self.size = len(nodes)
        self.indentation = indentation + 1
        self.nodes = {node["id"].split(".")[-1]: Node(node, self.indentation) for node in nodes}

    def __str__(self):
        indentation = self.indentation * "\t"
        nodes_str = ""

        for node in self.nodes:
            nodes_str += f"{indentation}{self.nodes[node]}"

        return f"\n{indentation}Size: {self.size}\n{indentation}Nodes:{nodes_str}"

    def get_nodes_type(self):
        return {n_id: node.type for n_id, node in self.nodes.items()}


class Edge:
    def __init__(self, edge, indentation):
        self.id = edge["id"].split(".")[-1]
        self.type = self.id.split("@")[0]
        self.node_in = edge["in"].split(".")[-1]
        self.node_out = edge["out"].split(".")[-1]
        self.indentation = indentation + 1

    def __str__(self):
        indentation = self.indentation * "\t"
        return f"\n{indentation}Edge id: {self.id}\n{indentation}Node in: {self.node_in}\n{indentation}Node out: {self.node_out}\n"


class Function:
    def __init__(self, function):
        self.name = function["function"]
        self.id = function["id"].split(".")[-1]
        self.indentation = 1
        self.ast = AST(function["AST"], self.indentation)

    def __str__(self):
        indentation = self.indentation * "\t"
        return f"{indentation}Function Name: {self.name}\n{indentation}Id: {self.id}\n{indentation}AST:{self.ast}"

    def get_nodes(self):
        return self.ast.nodes

    def get_nodes_types(self):
        return self.ast.get_nodes_type()


class Properties:
    def __init__(self, props, indentation):
        self.size = len(props)
        self.indentation = indentation + 1
        self.pairs = {prop["key"]: prop["value"] for prop in props}

    def __str__(self):
        indentation = self.indentation * "\t"
        string = ""

        for prop in self.pairs:
            string += f"\n{indentation}Property - {prop} : {self.pairs[prop]}"

        return f"{indentation}{string}\n"

    def code(self):
        if self.has_code():
            code = self.pairs["CODE"]
            if self.has_type() and self.get_type() != "ANY" and self.get_type() not in code:
                code = f"{self.get_type()} {code}"
            return code
        return None

    def get_type(self):
        return self.pairs.get("TYPE_FULL_NAME")

    def has_type(self):
        return "TYPE_FULL_NAME" in self.pairs

    def has_code(self):
        return "CODE" in self.pairs

    def line_number(self):
        return self.pairs["LINE_NUMBER"] if self.has_line_number() else None

    def has_line_number(self):
        return "LINE_NUMBER" in self.pairs

    def column_number(self):
        return self.pairs["COLUMN_NUMBER"] if self.has_column_number() else None

    def has_column_number(self):
        return "COLUMN_NUMBER" in self.pairs

    def get(self):
        return self.pairs

    def get_operator(self):
        value = self.pairs.get("METHOD_FULL_NAME")
        if value is None:
            return value
        if ("<operator>" in value) or ("<operators>" in value):
            return value.split(".")[-1]
        return None


node_labels = ["Block", "Call", "Comment", "ControlStructure", "File", "Identifier", "FieldIdentifier", "Literal",
               "Local", "Member", "MetaData", "Method", "MethodInst", "MethodParameterIn", "MethodParameterOut",
               "MethodReturn", "Namespace", "NamespaceBlock", "Return", "Type", "TypeDecl", "Unknown"]

operators = ['addition', 'addressOf', 'and', 'arithmeticShiftRight', 'assignment',
             'assignmentAnd', 'assignmentArithmeticShiftRight', 'assignmentDivision',
             'assignmentMinus', 'assignmentMultiplication', 'assignmentOr', 'assignmentPlus',
             'assignmentShiftLeft', 'assignmentXor', 'cast', 'conditionalExpression',
             'division', 'equals', 'fieldAccess', 'greaterEqualsThan', 'greaterThan',
             'indirectFieldAccess', 'indirectIndexAccess', 'indirection', 'lessEqualsThan',
             'lessThan', 'logicalAnd', 'logicalNot', 'logicalOr', 'minus', 'modulo', 'multiplication',
             'not', 'notEquals', 'or', 'postDecrement', 'plus', 'postIncrement', 'preDecrement',
             'preIncrement', 'shiftLeft', 'sizeOf', 'subtraction']

node_labels += operators

node_labels = {label: i for i, label in enumerate(node_labels)}

PRINT_PROPS = True


class Node:
    def __init__(self, node, indentation):
        self.id = node["id"].split(".")[-1]
        self.label = self.id.split("@")[0]
        self.indentation = indentation + 1
        self.properties = Properties(node["properties"], self.indentation)
        self.edges = {edge["id"].split(".")[-1]: Edge(edge, self.indentation) for edge in node["edges"]}
        self.order = None
        operator = self.properties.get_operator()
        self.label = operator if operator is not None else self.label
        self._set_type()

    def __str__(self):
        indentation = self.indentation * "\t"
        properties = f"{indentation}Properties: {self.properties}\n"
        edges_str = ""

        for edge in self.edges:
            edges_str += f"{self.edges[edge]}"

        return f"\n{indentation}Node id: {self.id}\n{properties if PRINT_PROPS else ''}{indentation}Edges: {edges_str}"

    def connections(self, connections, e_type):
        for e_id, edge in self.edges.items():
            if edge.type != e_type: continue

            if edge.node_in in connections["in"] and edge.node_in != self.id:
                connections["in"][self.id] = edge.node_in

            if edge.node_out in connections["out"] and edge.node_out != self.id:
                connections["out"][self.id] = edge.node_out

        return connections

    def has_code(self):
        return self.properties.has_code()

    def has_line_number(self):
        return self.properties.has_line_number()

    def get_code(self):
        return self.properties.code()

    def get_line_number(self):
        return self.properties.line_number()

    def get_column_number(self):
        return self.properties.column_number()

    def _set_type(self):
        # label = self.label if self.operator is None else self.operator
        self.type = node_labels.get(self.label)  # Label embedding

        if self.type is None:
            log_warning("node", f"LABEL {self.label} not in labels!")
            self.type = len(node_labels) + 1


class InputDataset(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.iloc[index].input

    def get_loader(self, batch_size, shuffle=True):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle)


class Metrics:
    def __init__(self, outs, labels):
        self.scores = outs
        self.labels = labels
        self.transform()
        print(self.predicts)

    def transform(self):
        self.series = pd.Series(self.scores)
        self.predicts = self.series.apply(lambda x: 1 if x >= 0.5 else 0)
        self.predicts.reset_index(drop=True, inplace=True)

    def __str__(self):
        confusion = confusion_matrix(y_true=self.labels, y_pred=self.predicts)
        tn, fp, fn, tp = confusion.ravel()
        string = f"\nConfusion matrix: \n"
        string += f"{confusion}\n"
        string += f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n"
        string += '\n'.join([name + ": " + str(metric) for name, metric in self().items()])
        return string

    def __call__(self):
        _metrics = {"Accuracy": metrics.accuracy_score(y_true=self.labels, y_pred=self.predicts),
                    "Precision": metrics.precision_score(y_true=self.labels, y_pred=self.predicts),
                    "Recall": metrics.recall_score(y_true=self.labels, y_pred=self.predicts),
                    "F-measure": metrics.f1_score(y_true=self.labels, y_pred=self.predicts),
                    "Precision-Recall AUC": metrics.average_precision_score(y_true=self.labels, y_score=self.scores),
                    "AUC": metrics.roc_auc_score(y_true=self.labels, y_score=self.scores),
                    "MCC": metrics.matthews_corrcoef(y_true=self.labels, y_pred=self.predicts),
                    "Error": self.error()}

        return _metrics

    def log(self):
        excluded = ["Precision-Recall AUC", "AUC"]
        _metrics = self()
        msg = ' - '.join(
            [f"({name[:3]} {round(metric, 3)})" for name, metric in _metrics.items() if name not in excluded])

        log_info('metrics', msg)

    def error(self):
        errors = [(abs(score - (1 if score >= 0.5 else 0)) / score) * 100 for score, label in
                  zip(self.scores, self.labels)]

        return sum(errors) / len(errors)


class Stat:
    def __init__(self, outs=None, loss=0.0, acc=0.0, labels=None):
        if labels is None:
            labels = []
        if outs is None:
            outs = []
        self.outs = outs
        self.labels = labels
        self.loss = loss
        self.acc = acc

    def __add__(self, other):
        return Stat(self.outs + other.outs, self.loss + other.loss, self.acc + other.acc, self.labels + other.labels)

    def __str__(self):
        return f"Loss: {round(self.loss, 4)}; Acc: {round(self.acc, 4)};"


@dataclass
class Stats:
    name: str
    results: List[Stat] = dataclasses.field(default_factory=list)
    total: Stat = Stat()

    def __call__(self, stat):
        self.total += stat
        self.results.append(stat)

    def __str__(self):
        return f"{self.name} {self.mean()}"

    def __len__(self):
        return len(self.results)

    def mean(self):
        res = Stat()
        res += self.total
        res.loss /= len(self)
        res.acc /= len(self)

        return res

    def loss(self):
        return self.mean().loss

    def acc(self):
        return self.mean().acc

    def outs(self):
        return self.total.outs

    def labels(self):
        return self.total.labels


if __name__ == '__main__':
    pass
