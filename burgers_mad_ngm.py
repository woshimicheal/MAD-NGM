"""
MAD-NGM example for the Burgers equation.

"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy import integrate
from scipy.io import loadmat

from MAD.Mad_pretain_burgers import MadNN, Torch_jax_gai
from NG.DNN_JAX_Burgers_1 import DNN
from NG.InitProb import Problem
from ops.Ops_Bugers import OpsBurgers


@dataclass
class BurgersData:
    xx_test: np.ndarray
    input_test: np.ndarray
    output_test: np.ndarray
    xx0: np.ndarray
    tt: np.ndarray


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_mad_layers(hidden_dim: int, num_layers: int, latent_dim: int) -> list[int]:
    input_dim = 1
    output_dim = 1
    layers = [input_dim] + [hidden_dim] * num_layers + [output_dim]
    layers[0] += layers[0] + latent_dim
    return layers


def load_burgers_data(data_path: str | Path, sub: int) -> BurgersData:
    data = loadmat(data_path)
    xx = data["xx"]
    tt = data["tspan"]
    input_data = data["input"]
    output_data = data["output"]

    xx_test = xx[:, ::sub]
    input_test = input_data[:, ::sub]
    output_test = output_data[:, :, ::sub]
    xx0 = xx_test.flatten()[:, None]

    return BurgersData(
        xx_test=xx_test,
        input_test=input_test,
        output_test=output_test,
        xx0=xx0,
        tt=tt,
    )


def default_pretrain_model_path(args: argparse.Namespace) -> Path:
    return args.output_dir / (
        f"Burgers_pretrain_lr{args.pretrain_lr}_iter{args.total_iterations}_"
        f"h{args.hidden_dim}_z{args.latent_dim}_n{args.num_latents}.pth"
    )


def default_pretrain_npz_path(args: argparse.Namespace) -> Path:
    return args.output_dir / (
        f"Burgers_pretrain_lr{args.pretrain_lr}_iter{args.total_iterations}_"
        f"h{args.hidden_dim}_z{args.latent_dim}_n{args.num_latents}.npz"
    )


def default_finetuned_model_path(args: argparse.Namespace) -> Path:
    return args.output_dir / f"Burgers_modefine_index{args.test_index}.pth"


def default_finetuned_npz_path(args: argparse.Namespace) -> Path:
    return args.output_dir / (
        f"Burgers_modefine_index{args.test_index}_h{args.hidden_dim}_"
        f"z{args.latent_dim}_n{args.num_latents}.npz"
    )


def default_evolution_path(args: argparse.Namespace) -> Path:
    return args.output_dir / (
        f"Burgers_time_evolution_{args.method_name}_index{args.test_index}_"
        f"h{args.hidden_dim}_z{args.latent_dim}_n{args.num_latents}.npz"
    )


def reset_solve_ivp_monitor() -> None:
    for name in ("last_log_time", "total_time"):
        if hasattr(monitor_ivp_progress, name):
            delattr(monitor_ivp_progress, name)


def monitor_ivp_progress(t, y, dt_log: float = 0.1):
    if not hasattr(monitor_ivp_progress, "last_log_time"):
        monitor_ivp_progress.last_log_time = -np.inf

    if not hasattr(monitor_ivp_progress, "total_time"):
        return None

    total_time = monitor_ivp_progress.total_time
    if total_time <= 0:
        return None

    if (t - monitor_ivp_progress.last_log_time >= dt_log) or (t >= total_time * 0.99):
        progress = t / total_time * 100
        print(f"Integration progress: {progress:.1f}% | t={t:.4f} / {total_time:.4f}")
        monitor_ivp_progress.last_log_time = t

    return None


class ProgressMonitor:
    def __init__(self, total_steps: int, description: str = "Progress"):
        self.total_steps = total_steps
        self.start_time = time.time()
        self.last_print_time = 0.0
        self.description = description

    def update(self, current_step: int) -> None:
        current_time = time.time()
        if (current_time - self.last_print_time > 1) or (current_step >= self.total_steps):
            progress = current_step / self.total_steps * 100
            elapsed_time = current_time - self.start_time
            remaining_time = (elapsed_time / (current_step + 1e-6)) * (self.total_steps - current_step)
            print(
                f"{self.description}: {progress:.1f}% | "
                f"elapsed={elapsed_time:.1f}s | remaining={remaining_time:.1f}s"
            )
            self.last_print_time = current_time


def pretrain_mad_ngm(args: argparse.Namespace, device: torch.device) -> tuple[Path, Path]:
    burgers_data = load_burgers_data(args.data_path, args.sub)
    layers = build_mad_layers(args.hidden_dim, args.num_layers, args.latent_dim)

    print(f"Final layers: {layers}")
    print(f"Training samples: {args.num_latents}")
    print(f"Spatial grid shape after downsampling: {burgers_data.xx0.shape}")

    u0_mat = torch.tensor(
        burgers_data.input_test[: args.num_latents, :],
        requires_grad=True,
        dtype=torch.float32,
        device=device,
    )

    mad = MadNN(
        u0Mat=u0_mat,
        xx0=burgers_data.xx0,
        layers=layers,
        latent_size=args.latent_dim,
        num_latents=args.num_latents,
        maxiter=args.lbfgs_max_iter,
        model="pretrain",
    )

    print("Start Burgers MAD-NGM pre-training...")
    start_time = time.perf_counter()

    while mad.iter < args.total_iterations:
        loss = mad.train()

        if mad.iter >= args.total_iterations:
            print(f"Pre-training finished. Total iterations: {mad.iter}")
            break

    total_time = time.perf_counter() - start_time
    print(f"Pre-training runtime: {total_time / 60:.2f} min")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.pretrained_model) if args.pretrained_model else default_pretrain_model_path(args)
    npz_path = Path(args.pretrained_npz) if args.pretrained_npz else default_pretrain_npz_path(args)

    torch.save(mad.net.state_dict(), model_path)
    latent_jax, init_az = Torch_jax_gai(str(model_path), n=args.torch_to_jax_n)
    jnp.savez(npz_path, Latent_jax=latent_jax, initAZ_test=init_az)

    print(f"Saved PyTorch pre-trained model to {model_path}")
    print(f"Saved JAX-compatible pre-trained parameters to {npz_path}")

    return model_path, npz_path


def find_closest_latent(input_train: torch.Tensor, input_target: torch.Tensor) -> int:
    mean_squared_diff = torch.mean((input_train - input_target).pow(2), dim=1)
    return int(torch.argmin(mean_squared_diff).item())


def modefine_mad_ngm(
    args: argparse.Namespace,
    device: torch.device,
    pretrain_model_path: Path,
) -> tuple[Path, Path, int]:

    burgers_data = load_burgers_data(args.data_path, args.sub)
    layers = build_mad_layers(args.hidden_dim, args.num_layers, args.latent_dim)

    index = args.test_index
    u0_mat = torch.tensor(
        burgers_data.input_test[: args.num_latents, :],
        requires_grad=True,
        dtype=torch.float32,
        device=device,
    )
    u1_mat = torch.tensor(
        burgers_data.input_test[index, :],
        dtype=torch.float32,
        device=device,
    )

    closest_index = find_closest_latent(u0_mat, u1_mat)
    print(f"Selected test index: {index}")
    print(f"Closest pre-training latent index: {closest_index}")

    u1_mat_column = u1_mat.unsqueeze(1)
    u1_mat_train = u1_mat_column.permute(1, 0)

    para_re = torch.load(pretrain_model_path, map_location=device)
    latents_define = para_re["Latents"]
    del para_re["Latents"]
    latent = latents_define[closest_index, :].unsqueeze(0)

    mad = MadNN(
        u1_mat_train,
        burgers_data.xx0,
        layers,
        args.latent_dim,
        1,
        1800,
        "modefine",
        para_dict=para_re,
        Latent=latent,
    )

    print("Start Burgers latent modefine stage...")
    start_time = time.perf_counter()
    while mad.iter < args.modefine_iterations:
        loss = mad.train()  
    elapsed_time = time.perf_counter() - start_time
    print(f"elapsed={elapsed_time / 60:.2f} min")

    model_path = Path(args.modefine_model) if args.modefine_model else default_finetuned_model_path(args)
    npz_path = Path(args.modefine_npz) if args.modefine_npz else default_finetuned_npz_path(args)

    torch.save(mad.net.state_dict(), model_path)
    latent_jax, init_az = Torch_jax_gai(str(model_path), n=args.torch_to_jax_n)
    jnp.savez(npz_path, Latent_jax=latent_jax, initAZ_test=init_az)

    print(f"Saved modefine PyTorch model to {model_path}")
    print(f"Saved modefine JAX-compatible parameters to {npz_path}")

    return model_path, npz_path, closest_index


def build_burgers_ops(args: argparse.Namespace):
    problem = Problem(args.problem_name, args.sample_name, args.hidden_dim, args.num_layers)
    dnn = DNN(
        args.activation,
        args.hidden_dim,
        args.num_layers,
        problem.dim,
        problem.Omega,
        args.latent_dim,
    )
    ops = OpsBurgers(problem, dnn, args.time_integrator, "NGLM")
    return problem, dnn, ops


def evolve_mad_ngm(args: argparse.Namespace, modefine_npz_path: Path) -> Path:
    os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

    burgers_data = load_burgers_data(args.data_path, args.sub)
    xx_test = jnp.array(burgers_data.xx_test.flatten()[:, None])
    input_test = jnp.array(burgers_data.input_test)
    output_test = jnp.array(burgers_data.output_test)
    delta = args.delta

    data = jnp.load(modefine_npz_path)
    latent_jax = data["Latent_jax"]
    init_az = data["initAZ_test"]

    if latent_jax.ndim > 1:
        latent_jax = latent_jax[0]

    start = timer()
    time_seed = datetime.now().timestamp() + (timer() - start)
    key = jax.random.PRNGKey(int(time_seed * 1000))

    _, dnn, ops = build_burgers_ops(args)

    prediction_initial = dnn.ufunLatent(xx_test, latent_jax, delta, init_az)
    input_define = input_test[args.test_index]
    initial_mse = jnp.mean(jnp.square(prediction_initial - input_define))
    print(f"Initial MSE before time evolution: {float(initial_mse):.6e}")

    def rhs_fun(t, alpha_z):
        return ops.rhsJ(xx_test, alpha_z, latent_jax, delta, t, key)

    store_indices = jnp.arange(0, args.num_time_steps + 1, 1)
    t_eval = args.max_step * store_indices

    print("Warming up JAX compilation...")
    warmup_out = rhs_fun(t_eval[0], init_az)
    jax.block_until_ready(warmup_out)
    print("Warm-up finished.")

    reset_solve_ivp_monitor() 
    monitor_ivp_progress.total_time = float(t_eval[-1] - t_eval[0])

    print(f"Start time evolution with {args.time_integrator}...")
    start_time = time.perf_counter()
    solution = integrate.solve_ivp(
        rhs_fun,
        [float(t_eval[0]), float(t_eval[-1])],
        np.asarray(init_az),
        method=args.time_integrator,
        t_eval=np.asarray(t_eval),
        jac=None,
        max_step=args.max_step,
        events=(lambda t, y: monitor_ivp_progress(t, y) or 0) if args.show_progress else None,
    )
    jax.block_until_ready(jnp.asarray(solution.y))
    elapsed_time = time.perf_counter() - start_time

    print(f"Time-evolution runtime: {elapsed_time:.2f} seconds")
    print(f"Time-evolution runtime: {elapsed_time / 60:.2f} minutes")

    evolution_path = Path(args.evolution_npz) if args.evolution_npz else default_evolution_path(args)
    jnp.savez(evolution_path, Lantent=latent_jax, az=solution.y)
    print(f"Saved time-evolution results to {evolution_path}")

    return evolution_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAD-NGM example for the Burgers equation.")

    parser.add_argument("--data_path", type=str, default="./Data/Burgers_small_demo.mat")
    parser.add_argument("--output_dir", type=Path, default=Path("./outputs/burgers"))
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "pretrain", "modefine", "evolve"],
        help="Which stage to run. 'modefine' corresponds to the notebook's modified/fine-tuning stage.",
    )

    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--pretrained_npz", type=str, default=None)
    parser.add_argument("--modefine_model", type=str, default=None)
    parser.add_argument("--modefine_npz", type=str, default=None)
    parser.add_argument("--evolution_npz", type=str, default=None)

    parser.add_argument("--seed", type=int, default=2345)
    parser.add_argument("--sub", type=int, default=2) 
    parser.add_argument("--hidden_dim", type=int, default=30)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--latent_dim", type=int, default=30)
    parser.add_argument("--num_latents", type=int, default=10)

    parser.add_argument("--pretrain_lr", type=float, default=0.01)
    parser.add_argument("--lbfgs_max_iter", type=int, default=100)
    parser.add_argument("--total_iterations", type=int, default=1000)
    parser.add_argument("--modefine_iterations", type=int, default=1800)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--torch_to_jax_n", type=int, default=4)

    parser.add_argument("--problem_name", type=str, default="Burgers")
    parser.add_argument("--sample_name", type=str, default="uni")
    parser.add_argument("--test_index", type=int, default=11)
    parser.add_argument("--delta", type=float, default=0.0) 

    parser.add_argument("--time_integrator", type=str, default="RK45")
    parser.add_argument("--max_step", type=float, default=1e-3)
    parser.add_argument("--num_time_steps", type=int, default=1000)
    parser.add_argument("--method_name", type=str, default="NGM")
    parser.add_argument("--show_progress", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    device = get_device()
    print(f"Using device: {device}")

    pretrain_model_path = Path(args.pretrained_model) if args.pretrained_model else None
    modefine_npz_path = Path(args.modefine_npz) if args.modefine_npz else None
    evolution_path = Path(args.evolution_npz) if args.evolution_npz else None

    if args.stage in {"all", "pretrain"}:
        pretrain_model_path, _ = pretrain_mad_ngm(args, device)

    if args.stage in {"all", "modefine"}:
        if pretrain_model_path is None:
            raise ValueError("Please provide --pretrained_model when running --stage modefine.")
        _, modefine_npz_path, _ = modefine_mad_ngm(args, device, pretrain_model_path)

    if args.stage in {"all", "evolve"}:
        if modefine_npz_path is None:
            raise ValueError("Please provide --modefine_npz when running --stage evolve.")
        evolution_path = evolve_mad_ngm(args, modefine_npz_path)


if __name__ == "__main__":
    main()
