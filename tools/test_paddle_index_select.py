import paddle

input = paddle.rand([2, 20, 2])  # [B, T, D]
index = paddle.to_tensor([[2, 3, 4], [3, 5, 6]])  # [B, T]

axis_b = paddle.arange(2)[:, None, None]
axis_t = paddle.arange(20)[None, :, None]
axis_d = paddle.arange(2)[None, None, :]
axis_index = index[:, :, None]
res = input[axis_b, axis_index, axis_d]

print(res)
