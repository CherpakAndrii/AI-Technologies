from models.Hamming import Hamming
from models.base_images import base_images


def calculate_hash(image: list[float]) -> int:
    return sum([bit*2**pos for pos, bit in enumerate(image)])


def get_copy_with_inverted_bit(img: list, idx: int) -> list[float]:
    new_img = []
    for i in range(len(img)):
        new_img.append(img[i] if i != idx else (1 if img[i] == 0 else 0))
    return new_img


def get_distorted_images(img: list, changed_bits: int, start_index: int = 0) -> list[list[float]]:
    if changed_bits > 1:
        for i in range(start_index, len(img)-changed_bits+1):
            new_base_img = get_copy_with_inverted_bit(img, i)
            for new_img in get_distorted_images(new_base_img, changed_bits-1, i+1):
                yield new_img
    else:
        for i in range(start_index, len(img)):
            yield get_copy_with_inverted_bit(img, i)


def make_test_data(length: int):
    test_data = []
    test_results = []

    hashes = set()

    for base_img in base_images:
        hashes.add(calculate_hash(base_img))

    for changed_bits in range(1, 3):
        local_test_data = []
        local_test_results = []

        local_hashes = set()
        excluded_hashes = set()

        for ind, img in enumerate(base_images):
            for new_img in get_distorted_images(img, changed_bits):
                img_hash = calculate_hash(new_img)
                if img_hash not in hashes:
                    if img_hash not in local_hashes:
                        local_test_data.append(new_img)
                        local_test_results.append(ind)
                        local_hashes.add(hash)
                    else:
                        excluded_hashes.add(img_hash)

        for i in range(len(local_test_data)):
            img_hash = calculate_hash(local_test_data[i])
            if img_hash not in excluded_hashes:
                test_data.append(local_test_data[i])
                test_results.append(local_test_results[i])
                hashes.add(img_hash)

    print(f"Data generated! Train: {len(base_images)}, test: {len(test_data)}")
    return (test_data, test_results)


if __name__ == '__main__':
    input_size = len(base_images[0])
    output_size = len(base_images)
    model = Hamming(input_size, output_size)
    test_data, test_results = make_test_data(input_size)
    model.train(base_images, [])

    correct, processed = model.test(test_data, test_results)
    print(f"Test results: {correct}/{processed}")

#    print(model.predict((
#        1.0, 0.0, 0.0, 1.0,
#        1.0, 0.0, 1.0, 1.0,
#        1.0, 1.0, 0.0, 1.0,
#        1.0, 0.0, 1.0, 1.0,
#        1.0, 0.0, 0.0, 1.0
# )))

