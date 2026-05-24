"""
MAD-NGM example for the Korteweg--de Vries (KdV) equation.

"""

from __future__ import annotations

import argparse
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
from scipy.io import loadmat

import matplotlib.pyplot as plt

from MAD.Mad_pretain_kdv import MadNN, Torch_jax_gai, data_loader
from NG.DNN_test_kdv import DNN
from NG.InitProb import Problem
from ops.OpsKdV import OpsKdV


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_mad_layers(hidden_dim: int, latent_dim: int) -> list[int]:
    input_dim = 1
    output_dim = 1
    return [2 * input_dim + latent_dim, hidden_dim, output_dim]


def pretrain_mad_ngm(args: argparse.Namespace, device: torch.device) -> Path:
    data = loadmat(args.data_path)
    xx = data["xx"]
    input_data = data["input"]
    xx0 = xx.flatten()[:, None]

    layers = build_mad_layers(args.hidden_dim, args.latent_dim)
    u0_mat = torch.tensor(
        input_data[: args.num_latents, :],
        requires_grad=True,
        dtype=torch.float32,
        device=device,
    )

    model = MadNN(
        u0_mat,
        xx0,
        layers,
        args.latent_dim,
        args.num_latents,
        args.lbfgs_max_iter,
        "pretrain",
    )

    print("Start MAD-NGM pre-training...")
    start_time = time.perf_counter()

    while model.iter < args.total_iterations:
        model.train()

    total_time = time.perf_counter() - start_time
    print(f"Pre-training finished in {total_time / 60:.2f} min.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / (
        f"kdv_pretrain_h{args.hidden_dim}_z{args.latent_dim}_n{args.num_latents}.pth"
    )
    parameter_path = args.output_dir / (
        f"kdv_pretrain_h{args.hidden_dim}_z{args.latent_dim}_n{args.num_latents}.npz"
    ) 
    

    torch.save(model.net.state_dict(), model_path)
    latent_jax, init_az = Torch_jax_gai(str(model_path), n=1)

    jnp.savez(parameter_path, Latent_jax=latent_jax, initAZ_test=init_az)

    print(f"Saved PyTorch checkpoint to {model_path}")
    print(f"Saved JAX-compatible parameters to {parameter_path}")

    return parameter_path


def loss(ufun, y, x, z, sigma=0.01):
    y_pred = ufun(x, z)
    mse = jnp.mean((y - y_pred)**2)
    latent_regularization = sigma * jnp.mean(jnp.square(z))
    return mse + latent_regularization
     

def fine_tune_latent(ufun, y, x, init_latent, max_iter: int, learning_rate: float, sigma: float = 0.01, print_every: int = 50):
    loss_fn = lambda z: loss(ufun, y, x, z, sigma)
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
            print(f"fine-tune iter={iteration:6d}, loss={loss_value:.6e}")

    final_loss = float(loss_fn(params))
    print(f"Fine-tuning finished. Final loss={final_loss:.6e}")
    return params

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
    ops = OpsKdV(problem, dnn, args.time_integrator, "NGLM")

    xx_test, input_test, _, output_test = data_loader(args.data_path, s=args.num_test_points)
    xx_train, input_train, _, _ = data_loader(args.data_path)

    input_define = input_test[args.test_index]
    input_train = input_train[: args.num_latents]

    data = np.load(parameter_path)
    latent_all = jnp.array(data["Latent_jax"])
    init_az = jnp.array(data["initAZ_test"])
    data.close()

    distances = jnp.linalg.norm(input_train - input_define, axis=1)
    closest_index = int(jnp.argmin(distances))
    latent = latent_all[closest_index]

    print(f"Selected test index: {args.test_index}")
    print(f"Closest training latent index: {closest_index}")

    ufun = lambda x, z: dnn.ufunLatent(x, z, init_az)
    initial_prediction = dnn.ufunLatent(xx_test, latent, init_az)
    initial_error = jnp.mean(jnp.square(initial_prediction - output_test[args.test_index, 0]))
    print(f"Initial MSE before fine-tuning: {float(initial_error):.6e}")

    latent = fine_tune_latent(
        ufun=ufun,
        y=input_define,
        x=xx_test,
        init_latent=latent,
        max_iter=args.finetune_iterations,
        learning_rate=args.finetune_lr,
    )

    fine_tuned_path = args.output_dir / f"kdv_finetuned_latent_index{args.test_index}.npz"
    jnp.savez(fine_tuned_path, Latent_jax=latent, initAZ_test=init_az)
    print(f"Saved fine-tuned latent representation to {fine_tuned_path}")

    def rhs_fun(t, alpha_z):
        return ops.rhsJ(xx_test, alpha_z, latent, t, key)

    store_indices = jnp.arange(0, args.num_time_steps + 1, 1)
    t_eval = args.max_step * store_indices

    print(f"Start time evolution with {args.time_integrator}...")
    start_time = time.perf_counter()
    solution = integrate.solve_ivp(
        rhs_fun,
        [float(t_eval[0]), float(t_eval[-1])],
        init_az,
        method=args.time_integrator,
        t_eval=t_eval,
        max_step=args.max_step,
    )
    runtime = time.perf_counter() - start_time
    print(f"Time-evolution runtime: {runtime / 60:.4f} min")

    evolution_path = args.output_dir / f"kdv_time_evolution_index{args.test_index}_{args.time_integrator}.npz"
    jnp.savez(evolution_path, Latent_jax=latent, initAZ_test=solution.y)
    print(f"Saved time-evolution results to {evolution_path}")

    if args.save_figure:
        figure_path = args.output_dir / f"kdv_prediction_index{args.test_index}_{args.time_integrator}.pdf"
        plot_solution_snapshots(
            dnn=dnn,
            x=xx_test,
            output_test=output_test,
            test_index=args.test_index,
            latent=latent,
            solution_y=solution.y,
            num_time_steps=args.num_time_steps,
            reference_time_scale=args.reference_time_scale,
            figure_path=figure_path,
        )


def plot_solution_snapshots(
    dnn,
    x,
    output_test,
    test_index: int,
    latent,
    solution_y,
    num_time_steps: int,
    reference_time_scale: int,
    figure_path: Path,
) -> None:
    time_stamps = [0.0, 0.2, 0.5, 1.0]

    plt.figure(figsize=(8, 6))
    for i, t_stamp in enumerate(time_stamps, start=1):
        parameter_index = int(t_stamp * num_time_steps)
        reference_index = int(t_stamp * reference_time_scale)

        prediction = dnn.ufunLatent(x, latent, solution_y[:, parameter_index])
        reference = output_test[test_index, reference_index]
        mse = float(jnp.mean(jnp.square(prediction - reference)))

        ax = plt.subplot(2, 2, i)
        ax.plot(x, reference, "r", label="Reference")
        ax.plot(x, prediction, "b--", label="Prediction")
        ax.set_title(f"t={t_stamp}, MSE={mse:.2e}")
        ax.set_xlabel("x")
        ax.set_xlim(-1, 1)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction figure to {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAD-NGM example for the KdV equation.")

    parser.add_argument("--data_path", type=str, default="./Data/kdv_small_demo.mat")
    parser.add_argument("--output_dir", type=Path, default=Path("./outputs/kdv"))
    parser.add_argument("--stage", type=str, default="all", choices=["all", "pretrain", "evaluate"])
    parser.add_argument("--pretrained_npz", type=str, default=None)

    parser.add_argument("--seed", type=int, default=2345)
    parser.add_argument("--hidden_dim", type=int, default=20)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--latent_dim", type=int, default=5)
    parser.add_argument("--num_latents", type=int, default=10) 

    parser.add_argument("--lbfgs_max_iter", type=int, default=20)
    parser.add_argument("--total_iterations", type=int, default=1000)
    parser.add_argument("--print_every", type=int, default=100)

    parser.add_argument("--problem_name", type=str, default="KdVTwoSol")
    parser.add_argument("--sample_name", type=str, default="uni")
    parser.add_argument("--test_index", type=int, default=11)
    parser.add_argument("--num_test_points", type=int, default=512)
    parser.add_argument("--finetune_iterations", type=int, default=400)
    parser.add_argument("--finetune_lr", type=float, default=0.01)

    parser.add_argument("--time_integrator", type=str, default="RK45")
    parser.add_argument("--max_step", type=float, default=1e-3)
    parser.add_argument("--num_time_steps", type=int, default=1000)
    parser.add_argument("--reference_time_scale", type=int, default=100)

    parser.add_argument("--save_figure", action="store_true")

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
