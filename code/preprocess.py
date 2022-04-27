
def flatten_and_label(data):
    import awkward as ak
    names = list(data.keys())
    flat_parts = {name: [ak.ravel(part) for part in data[name]]
                  for name in data}
    lens = tuple(len(part) for part in flat_parts[names[0]])
    # check all variables have the same length
    for name in names[1:]:
        lens_name = tuple(len(part) for part in flat_parts[name])
        assert lens == lens_name, f"Different lengths in {name}, {lens}, {lens_name}"
    data = {name: ak.concatenate(flat_parts[name]) for name in flat_parts}
    labels = ak.concatenate([[i]*l for i, l in enumerate(lens)])
    return labels, data


def get_weights(labels):
    import numpy as np
    weights = np.zeros(len(labels))
    for label in set(labels):
        mask = labels == label
        weights[mask] = 1/np.sum(mask)
    return weights


def split(labels, weights, data, splits=[0.2, 'remainder']):
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
        allocated_data = {name: values[mask] for name, values in
                          data.items()}
        parts.append((labels[mask], weights[mask], allocated_data))
    return parts

if __name__ == "__main__":
    import data_readers
    parts, data = data_readers.read()
    labels, data = flatten_and_label(data)
    weights = get_weights(labels)
    (test_labels, test_weights, test_data), \
            (train_labels, train_weight, train_data) =\
            split(labels, weights, data)


