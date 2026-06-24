import dataclasses
import logging
import os
import time

from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

from examples.aloha_real import env as _env


_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 25
    control_hz: float = 30.0

    num_episodes: int = 1
    max_episode_steps: int = 1000
    initial_reset: bool = True
    final_reset: bool = True


def _timing_log_chunks() -> int:
    return int(os.environ.get("OPENPI_RUNTIME_TIMING_LOG_STEPS", "8"))


def _run_episode(
    environment: _env.AlohaRealEnvironment,
    policy: _websocket_client_policy.WebsocketClientPolicy,
    args: Args,
    episode_index: int,
) -> None:
    logging.info("Starting episode %d", episode_index)
    print("[openpi] Starting episode", flush=True)
    if args.initial_reset:
        environment.reset()
    else:
        print("[openpi] Initial reset skipped; using current robot observation", flush=True)
        environment.reset(fake=True)
    observation = environment.get_observation()
    print("[openpi] Reset complete, requesting policy actions", flush=True)

    policy.reset()
    step_time = 1.0 / args.control_hz if args.control_hz > 0 else 0.0
    episode_steps = 0
    chunk_index = 0
    timing_log_chunks = _timing_log_chunks()

    while args.max_episode_steps <= 0 or episode_steps < args.max_episode_steps:
        chunk_start_time = time.monotonic()
        if chunk_index == 0:
            print("[openpi] Requesting first policy action chunk; server may compile on first call", flush=True)
        result = policy.infer(observation)
        action_time = time.monotonic()
        if "actions" not in result:
            raise RuntimeError(f"Policy result missing 'actions': keys={list(result.keys())}")

        actions = result["actions"]
        chunk_len = min(args.action_horizon, len(actions))
        if chunk_len <= 0:
            raise RuntimeError(f"Policy returned an empty action chunk: shape={getattr(actions, 'shape', None)}")
        if chunk_index == 0:
            print("[openpi] First policy action chunk received", flush=True)

        action_loop_start = time.monotonic()
        for action_index in range(chunk_len):
            if args.max_episode_steps > 0 and episode_steps >= args.max_episode_steps:
                break
            step_start_time = time.monotonic()
            environment.apply_action_open_loop({"actions": actions[action_index]})
            episode_steps += 1
            elapsed = time.monotonic() - step_start_time
            if step_time > 0 and elapsed < step_time:
                time.sleep(step_time - elapsed)

        action_loop_time = time.monotonic() - action_loop_start
        observation_start_time = time.monotonic()
        if args.max_episode_steps <= 0 or episode_steps < args.max_episode_steps:
            observation = environment.refresh_observation()
        observation_time = time.monotonic()

        if chunk_index < timing_log_chunks:
            print(
                "[openpi] Chunk timing "
                f"chunk={chunk_index} "
                f"steps={chunk_len} "
                f"policy_ms={(action_time - chunk_start_time) * 1000:.1f} "
                f"action_loop_ms={action_loop_time * 1000:.1f} "
                f"observe_ms={(observation_time - observation_start_time) * 1000:.1f} "
                f"total_steps={episode_steps}",
                flush=True,
            )

        if environment.is_episode_complete():
            break
        chunk_index += 1

    logging.info("Episode completed.")
    print("[openpi] Episode completed", flush=True)


def main(args: Args) -> None:
    logging.info("Connecting to policy server...")
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    metadata = ws_client_policy.get_server_metadata()
    logging.info("Server metadata: %s", metadata)
    logging.info("Creating ALOHA real environment...")
    reset_position = metadata.get("reset_pose")
    environment = _env.AlohaRealEnvironment(reset_position=reset_position)
    logging.info("ALOHA real environment ready")
    print("[openpi] ALOHA real environment ready", flush=True)

    try:
        for episode_index in range(args.num_episodes):
            _run_episode(environment, ws_client_policy, args, episode_index)
    finally:
        if args.final_reset:
            print("[openpi] Final reset", flush=True)
            environment.reset()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
