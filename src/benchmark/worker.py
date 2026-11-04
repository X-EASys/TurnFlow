"""Isolated native runtime for one source. Invoked by the public CLI."""
import argparse
import os
from pathlib import Path

from benchmark import dataset
from benchmark.io import read, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-key", required=True)
    args = parser.parse_args()
    directory = args.directory
    os.environ["BENCHMARK_LLM_ROUTES"] = str(directory / "native-routes.json")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    from loguru import logger
    logger.disable("benchmark")
    from benchmark.agent.base import AgentConfig
    from benchmark.agent.api_agent import APIAgent
    from benchmark.evaluator.task_entry import evaluate_simulation
    from benchmark.data_model.task import Task, EvaluationCriteria
    from benchmark.engine import run_single_scenario
    from benchmark.user.release_profile import create_release_simulator
    from benchmark.utils.llm_routes import reload_routes
    reload_routes()
    conf = dataset.protocol()
    path = dataset.source_path(args.source_key)
    scenario = dataset.load(args.source_key)
    simulator = create_release_simulator(
        scenario, conf["simulator_model"], source_path=path,
        generation_candidate=conf["generation_candidate"], candidate_source_key=args.source_key,
        candidate_bindings=dataset.bindings(), candidate_strict=True)
    agent = APIAgent(AgentConfig(name="turnflow-agent", model_id="turnflow-agent",
                                temperature=0.7, max_tokens=16384))
    simulation = run_single_scenario(
        scenario, agent, simulator, max_turns=conf["max_turns"],
        verbose=False, strict_calls=True, call_budget=simulator.trace.budget)
    task = Task(task_id=simulation.task_id, scenario=scenario,
                evaluation_criteria=EvaluationCriteria(max_turns=conf["max_turns"]))
    evaluation = evaluate_simulation(
        task, simulation, conf["judge_model"], source_path=path,
        scenario_key=args.source_key, runtime_profile=conf["judge_runtime_profile"])
    save(directory / "native-result.json", {
        "source_key": args.source_key, "simulation": simulation.model_dump(mode="json"),
        "evaluation": evaluation.model_dump(mode="json"),
        "decision_status": (evaluation.metadata or {}).get("decision_status", "INVALID"),
        "stages": simulator.trace.stages,
    })


if __name__ == "__main__":
    main()
