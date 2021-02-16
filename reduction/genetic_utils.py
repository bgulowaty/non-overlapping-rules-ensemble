import random
from typing import List
from deap import creator
from toolz.curried import pipe, map, reduce
import numpy as np

from .api import Rule, Statement


def get_rules_statements(rules):
    return pipe(
        rules,
        map(lambda r: list(r.statements)),
        reduce(list.__add__),
    )


def statements_individual_creator(rules: List[Rule]):
    statements = get_rules_statements(rules)

    statements_vector = pipe(
        statements,
        map(lambda s: s.threshold),
        list,
        np.array
    )
    return creator.Individual(statements_vector)


def statements_and_enablements_individual_creator(rules: List[Rule]):
    statements = get_rules_statements(rules)

    statements_vector = pipe(
        statements,
        map(lambda s: s.threshold),
        list,
        np.array
    )

    enablement_vector = [0 for _ in rules]

    return creator.Individual(np.concatenate((statements_vector, enablement_vector), axis=0))


def enablement_vector(statement: List[any]):
    vector = [0 for s in statement]

    return creator.Individual(vector)

def enablement_vector_mutation(vector, proba=0.1):
    flip = lambda val: 0 if val == 1 else 1

    return np.array([
        flip(item) if random.random() <= proba else item for item in vector
    ])


def mutate_only_statements_mutation(individual, single_statement_mutation):
    mutated_statements = mutate_all_statements(individual, single_statement_mutation)

    return (creator.Individual(mutated_statements),)

def mutate_all_statements(statements_vector, single_statement_mutation):
    statements_vector_iter = iter(statements_vector)
    mutated_statements = []

    for idx, current_val in enumerate(statements_vector_iter):
        next_val = next(statements_vector_iter)

        # shit
        if current_val <= next_val:
            new_current_val = single_statement_mutation(current_val)
            new_next_val = single_statement_mutation(next_val)
            while not new_current_val <= new_next_val:
                new_current_val = single_statement_mutation(current_val)
                new_next_val = single_statement_mutation(next_val)
            mutated_statements.append(new_current_val)
            mutated_statements.append(new_next_val)
        elif current_val > next_val:
            new_current_val = single_statement_mutation(current_val)
            new_next_val = single_statement_mutation(next_val)
            while not new_current_val > new_next_val:
                new_current_val = single_statement_mutation(current_val)
                new_next_val = single_statement_mutation(next_val)
            mutated_statements.append(new_current_val)
            mutated_statements.append(new_next_val)

    return mutated_statements

def statements_enablement_mutations(individual, statements_vector_size, single_statement_mutation,
                                    enablement_vector_mutation=enablement_vector_mutation):
    statements_vector = individual[:statements_vector_size]
    enablement_vector = individual[statements_vector_size:]

    mutated_statements = mutate_all_statements(statements_vector, single_statement_mutation)

    return (creator.Individual(
        np.concatenate((mutated_statements, enablement_vector_mutation(enablement_vector)), axis=0)),)


def statements_and_labels_individual_creator(statements: List[Statement], rules: List[Rule]):
    statements_vector = pipe(
        statements,
        map(lambda s: s.threshold),
        list,
        np.array
    )
    classes_vector = pipe(
        rules,
        map(lambda rule: rule.classified_class),
        list,
        np.array
    )

    return creator.Individual(np.concatenate((statements_vector, classes_vector), axis=0))


def label_vector_mutation(label_vector, available_labels, prob=0.5):
    new_label_vector = []

    for element in label_vector:
        if random.random() < prob:
            new_label_vector.append(np.random.choice(available_labels))
        else:
            new_label_vector.append(element)

    return new_label_vector


def mutation(individual, statements_vector_size, available_labels, single_statement_mutation, label_vector_mutation):
    statements_vector = individual[:statements_vector_size]
    lables_vector = individual[statements_vector_size:]

    mutated_statements = mutate_all_statements(statements_vector, single_statement_mutation)

    return (creator.Individual(
        np.concatenate((mutated_statements, label_vector_mutation(lables_vector, available_labels)), axis=0)),)
