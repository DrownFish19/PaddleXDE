import warnings

import paddle
import paddle.nn as nn

from ..utils.misc import flat_to_shape
from ..utils.ode_utils import _mixed_norm, _rms_norm
from .odeint import odeint


class OdeintAdjointMethod(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        func,
        y0,
        t,
        rtol,
        atol,
        method,
        options,
        event_fn,
        adjoint_rtol,
        adjoint_atol,
        adjoint_method,
        adjoint_options,
        t_requires_grad,
        *adjoint_params,
    ):

        ctx.func = func
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.t_requires_grad = t_requires_grad

        with paddle.no_grad():
            ans = odeint(
                func, y0, t, solver=method, rtol=rtol, atol=atol, options=options
            )
            ctx.save_for_backward(t, ans, *adjoint_params)

        return ans

    @staticmethod
    def backward(ctx, grad_y):
        """
        因为不包含event模式, ans仅为solution
        所以直接使用grad_y即可对应forward输出tensor的梯度
        :param ctx:
        :param grad_y:
        :return:
        """
        with paddle.no_grad():
            func = ctx.func
            adjoint_rtol = ctx.adjoint_rtol
            adjoint_atol = ctx.adjoint_atol
            adjoint_method = ctx.adjoint_method
            adjoint_options = ctx.adjoint_options
            t_requires_grad = ctx.t_requires_grad

            # Backprop as if integrating up to event time.
            # Does NOT backpropagate through the event time.

            t, y, *adjoint_params = ctx.saved_tensor()

            adjoint_params = tuple(adjoint_params)

            ##################################
            #      Set up initial state      #
            ##################################

            # [-1] because y and grad_y are both of shape (len(t), *y0.shape) 初始状态为最后一个时刻的数据和梯度
            aug_state = [
                paddle.zeros([], dtype=y.dtype),
                y[-1],
                grad_y[-1],
            ]  # vjp_t, y, vjp_y
            aug_state.extend(
                [paddle.zeros_like(param) for param in adjoint_params]
            )  # vjp_params

            ##################################
            #    Set up backward ODE func    #
            ##################################

            # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
            def augmented_dynamics(t, y_aug):
                # Dynamics of the original system augmented with
                # the adjoint wrt y, and an integrator wrt t and args.
                y = y_aug[1]
                adj_y = y_aug[2]
                # ignore gradients wrt time and parameters

                with paddle.set_grad_enabled(True):
                    t_ = t.detach()
                    t = paddle.assign(t)
                    t.stop_gradient = False
                    y = paddle.assign(y)
                    y.stop_gradient = False

                    # If using an adaptive solver we don't want to waste time resolving dL/dt unless we need it (which
                    # doesn't necessarily even exist if there is piecewise structure in time), so turning off gradients
                    # wrt t here means we won't compute that if we don't need it.
                    func_eval = func(t if t_requires_grad else t_, y)

                    vjp_t, vjp_y, *vjp_params = paddle.autograd.grad(
                        outputs=func_eval,
                        inputs=(t, y) + adjoint_params,
                        grad_outputs=-adj_y,
                        allow_unused=True,
                        retain_graph=True,
                    )

                # autograd.grad returns None if no gradient, set to zero.
                vjp_t = paddle.zeros_like(t) if vjp_t is None else vjp_t
                vjp_y = paddle.zeros_like(y) if vjp_y is None else vjp_y
                vjp_params = [
                    paddle.zeros_like(param) if vjp_param is None else vjp_param
                    for param, vjp_param in zip(adjoint_params, vjp_params)
                ]

                return vjp_t, func_eval, vjp_y, *vjp_params

            ##################################
            #       Solve adjoint ODE        #
            ##################################

            if t_requires_grad:
                time_vjps = paddle.empty([len(t)], dtype=t.dtype)
            else:
                time_vjps = None
            for i in range(len(t) - 1, 0, -1):
                if t_requires_grad:
                    # Compute the effect of moving the current time measurement point.
                    # We don't compute this unless we need to, to save some computation.
                    func_eval = func(t[i], y[i])
                    dLd_cur_t = func_eval.reshape(-1).dot(grad_y[i].reshape(-1))
                    aug_state[0] -= dLd_cur_t
                    time_vjps[i] = dLd_cur_t

                # Run the augmented system backwards in time.
                aug_state = odeint(
                    func=augmented_dynamics,
                    y0=tuple(aug_state),
                    t=t[i - 1 : i + 1].flip(0),
                    solver=adjoint_method,
                    rtol=adjoint_rtol,
                    atol=adjoint_atol,
                    options=adjoint_options,
                )
                aug_state = [a[1] for a in aug_state]  # extract just the t[i - 1] value
                aug_state[1] = y[
                    i - 1
                ]  # update to use our forward-pass estimate of the state
                aug_state[2] += grad_y[
                    i - 1
                ]  # update any gradients wrt state at this time point

            if t_requires_grad:
                time_vjps[0] = aug_state[0]

            # adj_y = aug_state[2]
            adj_params = aug_state[3:]

        return (None, time_vjps, *adj_params)


def odeint_adjoint(
    func: callable,
    y0,
    t,
    *,
    rtol=1e-7,
    atol=1e-9,
    solver=None,
    options={"norm": _rms_norm},
    event_fn=None,
    adjoint_rtol=None,
    adjoint_atol=None,
    adjoint_solver=None,
    adjoint_options=None,
    adjoint_params=None,
):
    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if adjoint_params is None and not isinstance(func, nn.Layer):
        raise ValueError(
            "func must be an instance of nn.Module to specify the adjoint parameters; alternatively they "
            "can be specified explicitly via the `adjoint_params` argument. If there are no parameters "
            "then it is allowable to set `adjoint_params=()`."
        )

    # Must come before _check_inputs as we don't want to use normalised input (in particular any changes to options)
    if adjoint_rtol is None:
        adjoint_rtol = rtol
    if adjoint_atol is None:
        adjoint_atol = atol
    if adjoint_solver is None:
        adjoint_solver = solver

    if adjoint_solver != solver and options is not None and adjoint_options is None:
        raise ValueError(
            "If `adjoint_method != method` then we cannot infer `adjoint_options` from `options`. So as "
            "`options` has been passed then `adjoint_options` must be passed as well."
        )

    if adjoint_options is None:
        adjoint_options = (
            {k: v for k, v in options.items() if k != "norm"}
            if options is not None
            else {}
        )
    else:
        # Avoid in-place modifying a user-specified dict.
        adjoint_options = adjoint_options.copy()

    if adjoint_params is None:
        adjoint_params = tuple(find_parameters(func))
    else:
        adjoint_params = tuple(adjoint_params)  # in case adjoint_params is a generator.

    # Filter params that don't require gradients.
    oldlen_ = len(adjoint_params)
    adjoint_params = tuple(p for p in adjoint_params if p.trainable)
    if len(adjoint_params) != oldlen_:
        # Some params were excluded.
        # Issue a warning if a user-specified norm is specified.
        if "norm" in adjoint_options and callable(adjoint_options["norm"]):
            warnings.warn(
                "An adjoint parameter was passed without requiring gradient. For efficiency this will be "
                "excluded from the adjoint pass, and will not appear as a tensor in the adjoint norm."
            )

    # Handle the adjoint norm function.
    state_norm = options["norm"]
    handle_adjoint_norm_(adjoint_options, None, state_norm)  # todo:shapes

    solution = OdeintAdjointMethod.apply(
        func,
        y0,
        t,
        rtol,
        atol,
        solver,
        options,
        event_fn,
        adjoint_rtol,
        adjoint_atol,
        adjoint_solver,
        adjoint_options,
        not t.stop_gradient,
        *adjoint_params,
    )  # todo:shapes

    return solution


def find_parameters(module):
    assert isinstance(module, nn.Layer)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, "_is_replica", False):

        def find_tensor_attributes(module):
            tuples = [
                (k, v)
                for k, v in module.__dict__.items()
                if paddle.is_tensor(v) and v.requires_grad
            ]  # todo  DataParallel
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())


def handle_adjoint_norm_(adjoint_options, shapes, state_norm):
    """In-place modifies the adjoint options to choose or wrap the norm function."""

    # This is the default adjoint norm on the backward pass: a mixed norm over the tuple of inputs.
    def default_adjoint_norm(tensor_tuple):
        t, y, adj_y, *adj_params = tensor_tuple
        # (If the state is actually a flattened tuple then this will be unpacked again in state_norm.)
        return max(t.abs(), state_norm(y), state_norm(adj_y), _mixed_norm(adj_params))

    if "norm" not in adjoint_options:
        # `adjoint_options` was not explicitly specified by the user. Use the default norm.
        adjoint_options["norm"] = default_adjoint_norm
    else:
        # `adjoint_options` was explicitly specified by the user...
        try:
            adjoint_norm = adjoint_options["norm"]
        except KeyError:
            # ...but they did not specify the norm argument. Back to plan A: use the default norm.
            adjoint_options["norm"] = default_adjoint_norm
        else:
            # ...and they did specify the norm argument.
            if adjoint_norm == "seminorm":
                # They told us they want to use seminorms. Slight modification to plan A: use the default norm,
                # but ignore the parameter state
                def adjoint_seminorm(tensor_tuple):
                    t, y, adj_y, *adj_params = tensor_tuple
                    # (If the state is actually a flattened tuple then this will be unpacked again in state_norm.)
                    return max(t.abs(), state_norm(y), state_norm(adj_y))

                adjoint_options["norm"] = adjoint_seminorm
            else:
                # And they're using their own custom norm.
                if shapes is None:
                    # The state on the forward pass was a tensor, not a tuple. We don't need to do anything, they're
                    # already going to get given the full adjoint state as (t, y, adj_y, adj_params)
                    pass  # this branch included for clarity
                else:
                    # This is the bit that is tuple/tensor abstraction-breaking, because the odeint machinery
                    # doesn't know about the tupled nature of the forward state. We need to tell the user's adjoint
                    # norm about that ourselves.

                    def _adjoint_norm(tensor_tuple):
                        t, y, adj_y, *adj_params = tensor_tuple
                        y = flat_to_shape(y, (), shapes)
                        adj_y = flat_to_shape(adj_y, (), shapes)
                        return adjoint_norm((t, *y, *adj_y, *adj_params))

                    adjoint_options["norm"] = _adjoint_norm
