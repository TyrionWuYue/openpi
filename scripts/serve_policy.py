import dataclasses
import enum
import logging
import pathlib
import socket
import time

import tyro

from openpi.policies import aloha_policy as _aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # Run dummy inference calls before listening for robot clients. This pays the JAX/XLA compile cost up front.
    # Set to 0 only when intentionally debugging server startup without compiling the policy first.
    warmup_steps: int = 2
    # Persistent JAX compilation cache directory. Set to an empty string to leave JAX defaults unchanged.
    jax_cache_dir: str = "~/.cache/jax"

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def _policy_config_name(args: Args) -> str:
    match args.policy:
        case Checkpoint():
            return args.policy.config
        case Default():
            return DEFAULT_CHECKPOINT[args.env].config


def _make_warmup_observation(args: Args) -> dict:
    config_name = _policy_config_name(args)
    if "aloha" in config_name:
        obs = _aloha_policy.make_aloha_example()
        if args.default_prompt is not None:
            obs["prompt"] = args.default_prompt
        return obs

    raise ValueError(
        f"Warmup observation is not implemented for config {config_name!r}. "
        "Set --warmup-steps=0 or add a matching dummy observation."
    )


def warmup_policy(policy: _policy.Policy, args: Args) -> None:
    if args.warmup_steps <= 0:
        logging.warning("Policy warmup is disabled; the first robot inference request may block on JAX/XLA compile.")
        return

    obs = _make_warmup_observation(args)
    logging.info("Warming up policy with %d dummy inference call(s) before opening the websocket server...", args.warmup_steps)
    for step in range(args.warmup_steps):
        start_time = time.monotonic()
        action = policy.infer(obs)
        elapsed = time.monotonic() - start_time
        logging.info(
            "Warmup %d/%d completed in %.1fs; actions_shape=%s",
            step + 1,
            args.warmup_steps,
            elapsed,
            getattr(action.get("actions"), "shape", None),
        )
    logging.info("Policy warmup complete; server is ready for robot clients.")


def main(args: Args) -> None:
    if args.jax_cache_dir:
        import jax

        jax_cache_dir = pathlib.Path(args.jax_cache_dir).expanduser()
        jax_cache_dir.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(jax_cache_dir))
        logging.info("Using JAX compilation cache: %s", jax_cache_dir)

    policy = create_policy(args)
    policy_metadata = policy.metadata
    warmup_policy(policy, args)

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
