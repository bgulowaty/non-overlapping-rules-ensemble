from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from itertools import groupby
from typing import Set, Dict, Tuple, List, Union

from attr import attrs, attrib
from toolz import pipe
from toolz.curried import filter, map


class Relation(Enum):
    LEQ = "<="
    MT = ">"


@attrs(auto_attribs=True, frozen=True)
class Feature:
    index: int
    name: str

    def get_index(self) -> int:
        return self.index

    def get_name(self) -> str:
        return self.name


@attrs(auto_attribs=True, hash=True, eq=True)
class Statement:
    feature_idx: int
    relation: Relation
    threshold: float

    def contains(self, value: float) -> bool:
        if self.relation is Relation.LEQ:
            return value <= self.threshold

        return value > self.threshold

    def covers(self, feature_idx: int, value: float) -> bool:
        return self.feature_idx == feature_idx and self.contains(value)

    def covers_sample(self, X_sample: List[float]) -> bool:
        return self.contains(X_sample[self.feature_idx])

    def is_inside(self, statement: Statement) -> bool:
        is_same_idx = self.feature_idx == statement.feature_idx

        return is_same_idx and self.contains(statement.threshold)

@attrs(auto_attribs=True, frozen=True)
class Instance:
    value_by_feature_idx: Dict[int, float]


@attrs(hash=False, eq=True)
class Rule:
    statements: List[Statement] = attrib(converter=tuple)
    distribution_or_class: Union[any, Dict[any, int]] = attrib()

    def __hash__(self) -> int:
        if isinstance(self.distribution_or_class, dict):
            return hash((self.statements, frozenset(self.distribution_or_class.items())))

        return hash((self.statements, self.distribution_or_class))

    @property
    def classified_class(self) -> any:
        if isinstance(self.distribution_or_class, dict):
            return max(self.distribution_or_class, key=self.distribution_or_class.get)
        return self.distribution_or_class

    def get_statements_by_feature(self) -> Dict[int, Set[Statement]]:
        return {feature: list(statements) for feature, statements in
                                       groupby(sorted(self.statements, key=lambda s: s.feature_idx), lambda statement: statement.feature_idx)}

    def get_features(self) -> Set[int]:
        return pipe(
            self.statements,
            map(lambda statement: statement.feature_idx),
            set
        )

    def get_statements_for_feature(self, feature_idx: int) -> Set[Statement]:
        return pipe(
            self.statements,
            filter(lambda statement: statement.feature_idx == feature_idx),
            set
        )

    def describes(self, instance: Instance) -> bool:
        each_statement_covers_any_instances_feature = True
        for statement in self.statements:
            statement_covers_any_instances_feature = any(
                [statement.covers(feature_idx, value) for feature_idx, value in instance.value_by_feature_idx.items()])
            each_statement_covers_any_instances_feature = False if statement_covers_any_instances_feature is False \
                else each_statement_covers_any_instances_feature
        return each_statement_covers_any_instances_feature


class DecisionTree(ABC):

    @abstractmethod
    def get_rules(self) -> Set[Rule]:
        raise NotImplementedError()


@attrs(auto_attribs=True, frozen=True)
class Features:
    _feature_name_by_index: Dict[int, str] = attrib()

    def __hash__(self) -> int:
        return hash(frozenset(self._feature_name_by_index))

    def get_by_index(self, index: int) -> Feature:
        return Feature(index, self._feature_name_by_index[index])

    def get_all(self) -> Tuple[Feature]:
        return tuple([Feature(index, feature) for index, feature in self._feature_name_by_index.items()])


class AdjacentOrNot(Enum):  # TODO(bgulowaty): replace with Literal[true,false] in Python 3.8
    ADJACENT = True,
    NOT_ADJACENT = False

    def as_number(self):
        return 1 if self == AdjacentOrNot.ADJACENT else 0


class RulesAdjacencyMeasurer(ABC):

    @abstractmethod
    def measure(self, rule1: Rule, rule2: Rule) -> AdjacentOrNot:
        raise NotImplementedError
