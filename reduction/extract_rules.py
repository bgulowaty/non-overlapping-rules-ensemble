from typing import Set, List

from sklearn.tree import _tree
from sklearn.tree._tree import Tree
from sklearn.tree import DecisionTreeClassifier
from attr import attrs, attrib

from .api import DecisionTree, Rule, Statement, Relation


@attrs(frozen=True, eq=False)
class SkLearnDecisionTree(DecisionTree):
    _skLearnTree: DecisionTreeClassifier = attrib()

    def get_rules(self) -> Set[Rule]:
        tree_: Tree = self._skLearnTree.tree_

        def recurse(node: int, statements: List[Statement] = []) -> Set[Rule]:
            currentNodeFeatureIndex: int = tree_.feature[node]

            if currentNodeFeatureIndex != _tree.TREE_UNDEFINED:
                threshold: float = tree_.threshold[node]

                leftStatements = statements + [Statement(currentNodeFeatureIndex, Relation.LEQ, threshold)]
                leftRules = recurse(tree_.children_left[node], leftStatements)

                rightStatements = statements + [Statement(currentNodeFeatureIndex, Relation.MT, threshold)]
                rightRules = recurse(tree_.children_right[node], rightStatements)
                return leftRules.union(rightRules)
            else:
                samplesCountForEachClass = tree_.value[node][0]
                samplesCountByClass = {self._skLearnTree.classes_[idx]: count for idx, count in
                                       enumerate(samplesCountForEachClass)}
                return {Rule(set(statements), samplesCountByClass)}

        return recurse(0)


def as_tree(clf: DecisionTreeClassifier) -> DecisionTree:
    return SkLearnDecisionTree(clf)


def extract_rules(clf: DecisionTreeClassifier) -> Set[Rule]:
    return as_tree(clf).get_rules()
