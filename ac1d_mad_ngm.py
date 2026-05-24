"""
MAD-NGM example for the one-dimensional Allen--Cahn equation.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from jax import grad, jit
from scipy import integrate

from MAD.Mad_pretrain_ac1d import MadNN, Mad_LoadData, Torch_jax_gai, data_loader1
from NG.DNN_JAX_AC import DNN
from NG.InitProb import Problem
from ops.Ops_AC import OpsAc

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_mad_layers(hidden_dim: int, latent_dim: int, num_layers: int = 2) -> list[int]:
    input_dim = 1
    output_dim = 1
    layers = [input_dim] + [hidden_dim] * num_layers + [output_dim]
    layers[0] = 2 * input_dim + latent_dim
    return layers


def pretrain_mad_ngm(args: argparse.Namespace, device: torch.device) -> Path:
    u_mat, xx0, delta = Mad_LoadData(args.data_path)
    layers = build_mad_layers(args.hidden_dim, args.latent_dim, args.num_layers)

    u0_mat = torch.tensor(
        u_mat[: args.num_latents, :],
        requires_grad=True,
        dtype=torch.float32,
        device=device,
    )

    model = MadNN(
        u0Mat=u0_mat,
        xx0=xx0,
        delta=delta,
        layers=layers,
        latent_size=args.latent_dim,
        num_latents=args.num_latents,
        maxiter=args.lbfgs_max_iter,
        model="pretrain",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / (
        f"ac1d_pretrain_h{args.hidden_dim}_z{args.latent_dim}_n{args.num_latents}.pth"
    )
    parameter_path = args.output_dir / (
        f"ac1d_pretrain_h{args.hidden_dim}_z{args.latent_dim}_n{args.num_latents}.npz"
    )

    print("Start MAD-NGM pre-training...")
    start_time = time.perf_counter()

    while model.iter < args.total_iterations:
        loss = model.train() 
        

    total_time = time.perf_counter() - start_time
    print(f"Pre-training finished in {total_time / 60:.2f} min.")

    torch.save(model.net.state_dict(), model_path)
    latent_jax, init_az = Torch_jax_gai(str(model_path), n=args.torch_to_jax_n)

    jnp.savez(parameter_path, Latent_jax=latent_jax, initAZ_test=init_az)

    print(f"Saved PyTorch checkpoint to {model_path}")
    print(f"Saved JAX-compatible parameters to {parameter_path}")

    return parameter_path


def _mse_residual(y: jnp.ndarray, prediction: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.square(y - prediction))


def fine_tune_loss(ufun, y, x, z, reg_weight = 0.05,):
    prediction = ufun(x, z)
    reconstruction_loss = _mse_residual(y, prediction)
    latent_regularization = reg_weight * jnp.mean(jnp.square(z))
    return reconstruction_loss + latent_regularization


def fine_tune_latent(
    ufun,
    y: jnp.ndarray,
    x: jnp.ndarray,
    init_latent: jnp.ndarray,
    max_iter: int,
    learning_rate: float,
    reg_weight: float = 0.05,
    print_every: int = 100,):
    loss_fn = lambda z: fine_tune_loss(
        ufun=ufun,
        y=y,
        x=x,
        z=z,
        reg_weight=reg_weight,
    )

    grad_fn = jit(lambda z: grad(loss_fn)(z))

    optimizer = optax.adam(learning_rate)
    params = init_latent
    opt_state = optimizer.init(params)

    for iteration in range(max_iter):
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if iteration % print_every == 0 or iteration == max_iter - 1:
            loss_value = float(loss_fn(params))
            print(
                f"fine-tune iter={iteration:6d}, "
                f"loss={loss_value:.6e}"
            )

    final_loss = float(loss_fn(params))
    final_mse = float(_mse_residual(y, ufun(x, params)))
    print(f"Final MSE={final_mse:.6e}")

    return params


def choose_initial_latent(
    latent_all: jnp.ndarray,
    input_train: jnp.ndarray,
    input_define: jnp.ndarray,
) -> tuple[jnp.ndarray, int]:
    distances = jnp.linalg.norm(input_train - input_define, axis=1)
    closest_index = int(jnp.argmin(distances))
    return latent_all[closest_index], closest_index


def maybe_report_ivp_progress(t: float, total_time: float, state: dict[str, float], dt_log: float = 0.1) -> None:
    last_log_time = state.get("last_log_time", -np.inf)
    if (t - last_log_time >= dt_log) or (t >= total_time * 0.99):
        progress = t / total_time * 100.0 if total_time > 0 else 100.0
        print(f"integration progress: {progress:.1f}% | t={t:.4f}/{total_time:.4f}")
        state["last_log_time"] = t


def evaluate_mad_ngm(args: argparse.Namespace, parameter_path: Path) -> None:
    start = timer()
    time_seed = datetime.now().timestamp() + (timer() - start)
    key = jax.random.PRNGKey(int(time_seed * 1000))

    problem = Problem(args.problem_name, args.sample_name, args.hidden_dim, args.num_layers)
    dnn = DNN(
        args.activation,
        args.hidden_dim,
        args.num_layers,
        problem.dim,
        problem.Omega,
        args.latent_dim,
    )
    ops = OpsAc(problem, dnn, args.time_integrator, "NGLM")

    xx_all, delta_all, input_all, output_all = data_loader1(args.data_path, s=args.num_test_points)

    input_define = jnp.array(input_all[args.test_index])
    input_train = jnp.array(input_all[: args.num_latents])
    x_test = jnp.array(xx_all[args.test_index])
    delta = jnp.array(delta_all[args.test_index])

    data = np.load(parameter_path)
    latent_all = jnp.array(data["Latent_jax"])
    init_az = jnp.array(data["initAZ_test"])
    data.close()

    latent, closest_index = choose_initial_latent(latent_all, input_train, input_define)

    print(f"Selected test index: {args.test_index}")
    print(f"Closest training latent index: {closest_index}")

    ufun = lambda x, z: dnn.ufunLatent(x, z, delta, init_az)
    initial_prediction = dnn.ufunLatent(x_test, latent, delta, init_az)
    initial_mse = float(jnp.mean(jnp.square(initial_prediction - input_define)))
    print(f"Initial MSE before fine-tuning: {initial_mse:.6e}")

    latent = fine_tune_latent(
        ufun=ufun,
        y=input_define,
        x=x_test,
        init_latent=latent,
        max_iter=args.finetune_iterations,
        learning_rate=args.finetune_lr,
        print_every=args.finetune_print_every,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fine_tuned_path = args.output_dir / f"ac1d_finetuned_latent_index{args.test_index}.npz"
    jnp.savez(fine_tuned_path, Latent_jax=latent, initAZ_test=init_az)
    print(f"Saved fine-tuned latent representation to {fine_tuned_path}")


    latent_lm = latent
    init_az_lm = init_az

    def rhs_fun(t, alpha_z):
        return ops.rhsJ(x_test, alpha_z, latent_lm, delta, t, key)

    store_indices = jnp.arange(0, args.num_time_steps + 1, 1)
    t_eval = args.max_step * store_indices
    t_eval_np = np.array(t_eval)
    init_az_np = np.array(init_az_lm)

    print(f"Start time evolution with {args.time_integrator}...")
    start_time = time.perf_counter()

    progress_state: dict[str, float] = {}
    total_time = float(t_eval_np[-1] - t_eval_np[0])

    solve_kwargs = {}
    if args.show_ivp_progress:
        solve_kwargs["events"] = lambda t, y: maybe_report_ivp_progress(t, total_time, progress_state) or 0

    solution = integrate.solve_ivp(
        rhs_fun,
        [float(t_eval_np[0]), float(t_eval_np[-1])],
        init_az_np,
        method=args.time_integrator,
        t_eval=t_eval_np,
        max_step=args.max_step,
        **solve_kwargs,
    )

    runtime = time.perf_counter() - start_time
    print(f"Time-evolution runtime: {runtime / 60:.4f} min")
    print(f"Time integrator success: {solution.success}; message: {solution.message}")

    evolution_path = args.output_dir / f"ac1d_time_evolution_index{args.test_index}_{args.time_integrator}.npz"
    jnp.savez(evolution_path, Latent_jax=latent_lm, initAZ_test=solution.y, t=solution.t)
    print(f"Saved time-evolution results to {evolution_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAD-NGM example for the one-dimensional Allen--Cahn equation.")

    parser.add_argument("--data_path", type=str, default="./Data/ac_small_demo.mat")
    parser.add_argument("--output_dir", type=Path, default=Path("./outputs/ac1d"))
    parser.add_argument("--stage", type=str, default="all", choices=["all", "pretrain", "evaluate"])
    parser.add_argument("--pretrained_npz", type=str, default=None)

    parser.add_argument("--seed", type=int, default=2345)
    parser.add_argument("--hidden_dim", type=int, default=30)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--latent_dim", type=int, default=30)
    parser.add_argument("--num_latents", type=int, default=10)

    parser.add_argument("--lbfgs_max_iter", type=int, default=10)
    parser.add_argument("--total_iterations", type=int, default=1000)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--torch_to_jax_n", type=int, default=2)

    parser.add_argument("--problem_name", type=str, default="AC") 
    parser.add_argument("--sample_name", type=str, default="uni")
    parser.add_argument("--test_index", type=int, default=11) 
    parser.add_argument("--num_test_points", type=int, default=512)
    parser.add_argument("--finetune_iterations", type=int, default=7000)
    parser.add_argument("--finetune_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_print_every", type=int, default=100)

    parser.add_argument(
        "--time_integrator",
        type=str,
        default="RK45",
    )
    parser.add_argument("--max_step", type=float, default=1e-3)
    parser.add_argument("--num_time_steps", type=int, default=2000)
    parser.add_argument("--reference_time_scale", type=int, default=100)

    parser.add_argument("--show_ivp_progress", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    torch.set_default_dtype(torch.float32)

    device = get_device()
    print(f"Using device: {device}")

    parameter_path = Path(args.pretrained_npz) if args.pretrained_npz else None

    if args.stage in {"all", "pretrain"}:
        parameter_path = pretrain_mad_ngm(args, device)

    if args.stage in {"all", "evaluate"}:
        if parameter_path is None:
            raise ValueError("Please provide --pretrained_npz when running with --stage evaluate.")
        evaluate_mad_ngm(args, parameter_path)


if __name__ == "__main__":
    main()

