# TODO: CNN Autograd Derivatives

The current CNN PINN path uses `torch.autograd.grad` on the full predicted grid.
For convolutional models, this is not the same as a pointwise spatial derivative.

The current pattern:

```python
du_dX = torch.autograd.grad(
    u,
    X,
    grad_outputs=torch.ones_like(u),
    retain_graph=True,
    create_graph=True,
)[0]
```

computes the derivative of the sum of all output pixels with respect to the
input grid. Because each convolutional output depends on neighboring input
pixels, this mixes cross-pixel Jacobian terms and should not be interpreted as
local `u_x`, `u_y`, `v_x`, or `v_y`.

Future work:

- Decide whether CNN physics derivatives should use finite differences on the
  predicted field instead of autograd.
- If true autograd derivatives are needed, compute selected Jacobian entries
  for sampled output pixels explicitly, e.g. `d u[t,i,j] / d X[t,i,j,0]`.
- Consider a hybrid architecture: CNN encoder for context plus coordinate MLP
  decoder for queried `(x, y, t)` points. This keeps CNN features while making
  pointwise autograd derivatives well-defined.
- Add tests that distinguish pointwise derivatives from full-grid summed
  gradients for convolutional models.

Keep the current implementation unchanged until this behavior is intentionally
redesigned.
