"""
MAD-NGM example for the two-dimensional Allen--Cahn equation.
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

from MAD.Mad_pretrain_ac2d import MadNN, Torch_jax_gai, heat_loadData
from NG.DNN_JAX_AC2d import DNN
from NG.InitProb import Problem
from ops.Ops_Heat import OpsHeat


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_mad_layers(hidden_dim: int, latent_dim: int, num_layers: int) -> list[int]:
    input_dim = 2
    output_dim = 1
    layers = [input_dim] + [hidden_dim] * num_layers + [output_dim]
    layers[0] = 2 * input_dim + latent_dim
    return layers



def pretrain_mad_ngm(args: argparse.Namespace, device: torch.device) -> Path:
    u_mat, xx0 = heat_loadData(args.data_path)
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
        layers=layers,
        latent_size=args.latent_dim,
        num_latents=args.num_latents,
        maxiter=args.lbfgs_max_iter,
        model="pretrain",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / (
        f"ac2d_pretrain_h{args.hidden_dim}_z{args.latent_dim}_n{args.num_latents}.pth"
    )
    parameter_path = args.output_dir / (
        f"ac2d_pretrain_h{args.hidden_dim}_z{args.latent_dim}_n{args.num_latents}.npz"
    )

    print("Start MAD-NGM pre-training...")
    start_time = time.perf_counter()

    while model.iter < args.total_iterations:
        model.train()

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
    reconstruction_loss = _mse_residual(y, ufun(x, z))
    latent_regularization = reg_weight * jnp.mean(jnp.square(z))
    return reconstruction_loss + latent_regularization


def fine_tune_latent(ufun, y, x, init_latent, max_iter: int, learning_rate: float, reg_weight: float = 0.05, print_every: int = 100):
    loss_fn = lambda z: fine_tune_loss(ufun, y, x, z, reg_weight=reg_weight)
    grad_fn = jit(lambda z: grad(loss_fn)(z))

    params = init_latent
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    for iteration in range(max_iter):
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if iteration % print_every == 0 or iteration == max_iter - 1:
            loss_value = float(loss_fn(params))
            print(f"fine-tune iter={iteration:6d}, loss={loss_value:.6e}")

    final_loss = float(loss_fn(params))
    final_mse = float(_mse_residual(y, ufun(x, params)))
    print(f"Fine-tuning finished. Final loss={final_loss:.6e}")
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
    ops = OpsHeat(problem, dnn, args.time_integrator, "NGLM")

    input_all, x_test = heat_loadData(args.data_path)
    input_define = jnp.array(input_all[args.test_index])
    input_train = jnp.array(input_all[: args.num_latents])
    x_test = jnp.array(x_test)

    data = np.load(parameter_path)
    latent_all = jnp.array(data["Latent_jax"])
    init_az = jnp.array(data["initAZ_test"])
    data.close()

    latent, closest_index = choose_initial_latent(latent_all, input_train, input_define)

    print(f"Selected test index: {args.test_index}")
    print(f"Closest training latent index: {closest_index}")

    ufun = lambda x, z: dnn.ufunLatent(x, z, args.delta, init_az)
    initial_prediction = dnn.ufunLatent(x_test, latent, args.delta, init_az)
    initial_mse = float(_mse_residual(input_define, initial_prediction))
    print(f"Initial MSE before fine-tuning: {initial_mse:.6e}")

    start_time = time.perf_counter()
    latent = fine_tune_latent(
        ufun=ufun,
        y=input_define,
        x=x_test,
        init_latent=latent,
        max_iter=args.finetune_iterations,
        learning_rate=args.finetune_lr,
        reg_weight=args.latent_reg,
        print_every=args.finetune_print_every,
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Fine-tuning runtime: {elapsed_time / 60:.2f} min")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fine_tuned_path = args.output_dir / f"ac2d_finetuned_latent_index{args.test_index}.npz"
    jnp.savez(fine_tuned_path, Latent_jax=latent, initAZ_test=init_az)
    print(f"Saved fine-tuned latent representation to {fine_tuned_path}")

    # =========================
    # Time evolution
    # =========================
    latent_lm = latent
    init_az_lm = init_az

    store_indices = jnp.arange(0, args.num_time_steps + 1, 1)
    t_eval = args.max_step * store_indices

    def rhs_fun(t, alpha_z):
        return ops.rhsJ(x_test, alpha_z, latent_lm, t, key)

    print(f"Start time evolution with {args.time_integrator}...")
    start_time = time.perf_counter()
    solution = integrate.solve_ivp(
        rhs_fun,
        [float(t_eval[0]), float(t_eval[-1])],
        init_az_lm,
        t_eval=t_eval,
        method=args.time_integrator,
    ) 
    runtime = time.perf_counter() - start_time
    print(f"Time-evolution runtime: {runtime / 60:.4f} min")

    evolution_path = args.output_dir / (
        f"ac2d_time_evolution_index{args.test_index}_{args.time_integrator}.npz")
    jnp.savez(evolution_path, Lantent=latent_lm, az=solution.y,)
    print(f"Saved time-evolution results to {evolution_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MAD-NGM open-source example for a two-dimensional Allen--Cahn equation."
    )

    parser.add_argument("--data_path", type=str, default="./Data/ac2d_small_demo.mat")
    parser.add_argument("--output_dir", type=Path, default=Path("./outputs/ac2d"))
    parser.add_argument("--stage", type=str, default="all", choices=["all", "pretrain", "evaluate"])
    parser.add_argument("--pretrained_npz", type=str, default=None)

    parser.add_argument("--seed", type=int, default=2345)
    parser.add_argument("--hidden_dim", type=int, default=30)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--latent_dim", type=int, default=60)
    parser.add_argument("--num_latents", type=int, default=10)

    parser.add_argument("--lbfgs_max_iter", type=int, default=100)
    parser.add_argument("--total_iterations", type=int, default=1000)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--torch_to_jax_n", type=int, default=8)

    parser.add_argument("--problem_name", type=str, default="Heat")
    parser.add_argument("--sample_name", type=str, default="uni")
    parser.add_argument("--test_index", type=int, default=13)
    parser.add_argument("--delta", type=float, default=0.0)

    parser.add_argument("--finetune_iterations", type=int, default=1000)
    parser.add_argument("--finetune_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_print_every", type=int, default=100)
    parser.add_argument("--latent_reg", type=float, default=0.05)

    parser.add_argument(
        "--time_integrator",
        type=str,
        default="RK45",
    )
    parser.add_argument("--max_step", type=float, default=1e-3)
    parser.add_argument("--num_time_steps", type=int, default=2000)
    parser.add_argument("--skip_time_evolution", action="store_true")

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
