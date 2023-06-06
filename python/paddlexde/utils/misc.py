def flat_to_shape(tensor, length, shapes, numels):
    tensor_list = []
    total = 0

    for shape, num_ele in zip(shapes, numels):
        next_total = total + num_ele
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        if len(shape) == 0:
            tensor_list.append(tensor[..., total:next_total].reshape((*length, 0)))
        else:
            tensor_list.append(tensor[..., total:next_total].reshape((*length, *shape)))
        total = next_total
    return tuple(tensor_list)
