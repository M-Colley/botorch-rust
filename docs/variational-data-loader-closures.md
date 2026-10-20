# Native external-data variational loss closures

`get_loss_closure(mll, data_loader=...)` and
`get_loss_closure_with_grads(mll, parameters, data_loader=...)` now support
native variational ELBOs. Each call consumes one actual minibatch, restarts
the DataLoader at the next epoch, and evaluates that batch's ELBO in Rust.
The likelihood uses the minibatch size; KL/prior terms retain `num_data`
and `beta` scaling. Stored training inputs and targets are not replaced.

The gradient closure calls the same native ELBO algorithm with analytic
gradients for the supplied batch. It returns selected parameters in the
requested order, including noise, inducing locations, kernel/mean parameters,
and variational means/Cholesky factors. `NdarrayOptimizationClosure` retains
its real native parameter-state updates with these external-data closures.
No reference BoTorch model is called by the implementation.

As with GPyTorch's inner variational model, DataLoader inputs and targets
must already be transformed into the model's training space. A batch is a
sequence of input tensors followed by targets. Single- and multiple-output
models, unequal final batch sizes, and epoch restarts are covered by live
parity against official BoTorch revision
`b0dd01be61d517c387e234903a6e2cc38c057ea2`. Native finite differences check
every raw parameter on a separate external dataset.

Exact GP MLL external-data closures remain explicitly unsupported: their
native likelihood is tied to their stored training-data covariance. This
does not add minibatch support to exact GP fitting. Uncombined ELBO terms
and unsupported closure keywords reject explicitly. The native default
variational fitting convenience routines still use their existing full-data
closure; supplying a custom closure to those convenience routines remains
an explicit unsupported configuration. Use the loss/gradient closure and
its state adapter directly for external-data evaluation.

Two port bugs were corrected: internal variational gradient closures now
call the ELBO gradient rather than an unavailable exact-MLL method; native
gradient closures now reject unsupported keywords instead of silently
ignoring them.

The official fallback's `chain.from_iterable` over infinitely repeated empty
DataLoader epochs loops forever when the loader yields no batches. The port
detects an empty epoch and raises `ValueError`. A bounded regression safely
reproduces the upstream repeated-empty-epoch loop without hanging a process.
The proposed upstream correction is
[patches/botorch-empty-data-loader.patch](patches/botorch-empty-data-loader.patch),
which restarts the iterator once and rejects a second empty iteration.
