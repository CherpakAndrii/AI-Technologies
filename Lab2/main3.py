from itertools import product
from random import randint

from models.Parceptron import Parceptron
from models.SingleNeuronParceptron import SingleNeuronParceptron

base_img = [
    1,1,1,1,
    1,0,0,0,
    1,1,1,1,
    1,0,0,0,
    1,1,1,1
]


def showing_func(input_img: tuple[float]):
    threshold = len(input_img) * 0.25
    delta = calculate_delta(input_img)
    return 1 if delta <= threshold else 0


def calculate_delta(img: tuple[float]):
    return sum([0 if el1 == el2 else 1 for el1, el2 in zip(img, base_img)])


def make_test_data(length: int):
    test_data = []
    test_results = []
    train_data = []
    train_results = []
    combinations = list(product([0, 1], repeat=length))
    sorted_combinations = sorted(combinations, key=lambda img: calculate_delta(img))
    selected_data = sorted_combinations[:30000] + [
        sorted_combinations[randint(30000, len(sorted_combinations) - 1)] for _ in range(10000)
    ]
    for combination in selected_data:
        y = showing_func(combination)
        random_number = randint(0, 9)
        if random_number >= 8:
            test_data.append(combination)
            test_results.append(y)
        else:
            train_data.append(combination)
            train_results.append(y)

    shuffle(train_data, train_results)
    print(f"Data generated! Train: {len(train_data)}, test: {len(test_data)}")
    return (test_data, test_results, train_data, train_results, sorted_combinations)


def shuffle(train_x, train_y):
    for _ in range(len(train_x)):
        a = randint(0, len(train_x)-1)
        b = randint(0, len(train_x)-1)
        train_x[a], train_x[b], train_y[a], train_y[b] = train_x[b], train_x[a], train_y[b], train_y[a]


if __name__ == '__main__':
    input_size = len(base_img)
    model = SingleNeuronParceptron(input_size)
    test_data, test_results, train_data, train_results, all_combinations = make_test_data(input_size)
    model.train(train_data, train_results, 10, 0.001)

    # correct, processed = model.test(test_data, test_results)
    correct, processed = model.test(all_combinations, [showing_func(combination) for combination in all_combinations])
    print(f"Test results: {correct}/{processed}")

