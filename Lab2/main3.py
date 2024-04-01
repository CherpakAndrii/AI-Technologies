from itertools import product
from random import randint

from models.Parceptron import Parceptron
from models.SingleNeuronParceptron import SingleNeuronParceptron
from models.base_images import base_images


def showing_func(input_img: tuple[float], base_img: tuple[float]):
    threshold = len(input_img) * 0.25
    delta = calculate_delta(input_img, base_img)
    return 1 if delta <= threshold else 0


def calculate_delta(img: tuple[float], base_img: tuple[float]) -> float:
    return sum([0 if el1 == el2 else 1 for el1, el2 in zip(img, base_img)])


def make_test_data(length: int):
    test_data = []
    test_results = []
    train_data = []
    train_results = []
    combinations = list(product([0, 1], repeat=length))
    sorted_combinations = sorted(combinations, key=lambda img: min([calculate_delta(img, base_img) for base_img in base_images]))
    print(len([img for img in sorted_combinations if min([calculate_delta(img, base_img) for base_img in base_images]) <= len(img) * 0.25]))
    selected_data = sorted_combinations[:110000] + [
        sorted_combinations[randint(30000, len(sorted_combinations) - 1)] for _ in range(20000)
    ]
    for combination in selected_data:
        y = [showing_func(combination, base_img) for base_img in base_images]
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
    input_size = len(base_images[0])
    output_size = len(base_images)
    # model = Parceptron(input_size, 1)
    model = SingleNeuronParceptron(input_size, output_size)
    test_data, test_results, train_data, train_results, all_combinations = make_test_data(input_size)
    model.train(train_data, train_results, 10, 0.01)

    # correct, processed = model.test(test_data, test_results)
    correct, processed = model.test(all_combinations, [[showing_func(combination, base_img) for base_img in base_images] for combination in all_combinations])
    print(f"Test results: {correct}/{processed}")

