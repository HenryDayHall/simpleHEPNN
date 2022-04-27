
def flatten_and_label(data):
    import awkward as ak
    import numpy as np
    names = list(data.keys())
    flat_parts = {name: [ak.ravel(part) for part in data[name]]
                  for name in data}
    lens = tuple(len(part) for part in flat_parts[names[0]])
    # check all variables have the same length
    for name in names[1:]:
        lens_name = tuple(len(part) for part in flat_parts[name])
        assert lens == lens_name, f"Different lengths in {name}, {lens}, {lens_name}"
    attribute_names = sorted(flat_parts.keys())
    inputs = np.hstack([ak.to_numpy(ak.concatenate(flat_parts[name])
                                    ).astype(float).reshape((-1, 1))
                        for name in attribute_names])
    labels = ak.concatenate([[i]*l for i, l in enumerate(lens)])
    return attribute_names, labels, inputs


def get_weights(labels):
    import numpy as np
    weights = np.zeros(len(labels))
    for label in set(labels):
        mask = labels == label
        weights[mask] = 1/np.sum(mask)
    return weights


def split(labels, weights, inputs, splits=[0.2, 'remainder']):
    import numpy as np
    total = len(labels)
    num_in_split = [int(total*s) if not s == 'remainder' else None
                    for s in splits]

    if None in num_in_split:
        # calculate the remaineder
        specified = sum(x for x in num_in_split if x != None)
        assert specified <= total
        remainder = total - specified
        num_in_split[num_in_split.index(None)] = remainder
    else:
        # the sum of the number in the splits must equal the total
        missing = total - sum(num_in_split)
        num_in_split[0] += missing

    assert sum(num_in_split) == total

    allocations = np.concatenate([[i]*n for i, n in
                                  enumerate(num_in_split)])
    np.random.shuffle(allocations)
    parts = []
    for a in range(len(splits)):
        mask = allocations == a
        parts.append((labels[mask], weights[mask], inputs[mask]))
    return parts


def make_test_train(data):
    attribute_names, labels, inputs = flatten_and_label(data)
    weights = get_weights(labels)
    (test_labels, test_weights, test_inputs), \
            (train_labels, train_weight, train_inputs) =\
            split(labels, weights, inputs)
    return attribute_names, \
            (test_labels, test_weights, test_inputs), \
            (train_labels, train_weight, train_inputs)


if __name__ == "__main__":
    import data_readers
    parts, data = data_readers.read()
    attribute_names, labels, inputs = flatten_and_label(data)
    weights = get_weights(labels)
    (test_labels, test_weights, test_inputs), \
            (train_labels, train_weight, train_inputs) =\
            split(labels, weights, inputs)


