"""
PHFE CLI - Command-line interface for Posthumanity's First Exam

Commands:
- task: Task queue operations for worker coordination
- benchmark: Load and inspect canonical benchmarks
- icr: Transform benchmarks to ICR format
- corpus: Generate answer key corpus
- distill: Run distillation training
- arxiv: Extract training data from arxiv papers
- curriculum: Generate curriculum problems
- eval: Run language competence evaluations
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="phfe",
    help="Posthumanity's First Exam - Benchmark suite for in-context learning",
    no_args_is_help=True,
)
console = Console()


# Subcommand groups
task_app = typer.Typer(help="Task queue operations for worker coordination")
benchmark_app = typer.Typer(help="Benchmark operations")
icr_app = typer.Typer(help="ICR transformation operations")
corpus_app = typer.Typer(help="Answer key corpus operations")
distill_app = typer.Typer(help="Distillation training operations")
arxiv_app = typer.Typer(help="arXiv pipeline operations")
curriculum_app = typer.Typer(help="Curriculum generation operations")
eval_app = typer.Typer(help="Language evaluation operations")

app.add_typer(task_app, name="task")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(icr_app, name="icr")
app.add_typer(corpus_app, name="corpus")
app.add_typer(distill_app, name="distill")
app.add_typer(arxiv_app, name="arxiv")
app.add_typer(curriculum_app, name="curriculum")
app.add_typer(eval_app, name="eval")


@app.command()
def version():
    """Show version information."""
    from phfe import __version__

    console.print(f"PHFE version {__version__}")


@app.command()
def status():
    """Show project status and configuration."""
    table = Table(title="PHFE Project Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    # Check what's configured
    components = [
        ("Orchestrator", "Ready", "Tutor caller and observability"),
        ("Benchmark", "Stub", "Needs benchmark data loading"),
        ("ICR Transform", "Stub", "Needs tutor integration"),
        ("Answer Key Corpus", "Stub", "Needs generation pipeline"),
        ("Distillation", "Stub", "Needs training loop"),
        ("arXiv Pipeline", "Stub", "Needs LaTeX tooling"),
        ("Curriculum", "Stub", "Needs problem generators"),
        ("Language Evals", "Partial", "Repetition eval implemented"),
    ]

    for name, status, details in components:
        table.add_row(name, status, details)

    console.print(table)


# === Task Queue Commands ===

# Default storage directory for task queue
DEFAULT_TASK_STORAGE = Path("./orchestrator_data")


def _get_task_queue(storage_dir: Optional[Path] = None) -> "TaskQueue":
    """Get task queue instance with storage."""
    from phfe.orchestrator import TaskQueue

    path = storage_dir or DEFAULT_TASK_STORAGE
    queue = TaskQueue(storage_dir=path)
    queue.load_from_storage()
    return queue


@task_app.command("report")
def task_report(
    storage_dir: Optional[Path] = typer.Option(None, help="Task storage directory"),
):
    """Show compact task queue status (for orchestration)."""
    queue = _get_task_queue(storage_dir)
    report = queue.get_compact_report()
    console.print(report)


@task_app.command("status")
def task_status(
    queue_name: Optional[str] = typer.Argument(None, help="Queue name (optional)"),
    storage_dir: Optional[Path] = typer.Option(None, help="Task storage directory"),
):
    """Show detailed task queue status."""
    from phfe.orchestrator import QueueType

    queue = _get_task_queue(storage_dir)

    if queue_name:
        try:
            q = QueueType(queue_name)
            statuses = {q: queue.get_queue_status(q)}
        except ValueError:
            console.print(f"[red]Unknown queue: {queue_name}[/red]")
            console.print(f"Available: {', '.join(q.value for q in QueueType)}")
            raise typer.Exit(1)
    else:
        statuses = queue.get_all_status()

    table = Table(title="Task Queue Status")
    table.add_column("Queue", style="cyan")
    table.add_column("Total")
    table.add_column("Pending", style="yellow")
    table.add_column("Claimed", style="blue")
    table.add_column("Done", style="green")
    table.add_column("Failed", style="red")
    table.add_column("Review", style="magenta")

    for q, s in statuses.items():
        table.add_row(
            q.value,
            str(s.total),
            str(s.pending),
            str(s.claimed),
            str(s.completed),
            str(s.failed),
            str(s.needs_review),
        )

    console.print(table)


@task_app.command("enqueue")
def task_enqueue(
    queue_name: str = typer.Argument(..., help="Queue name"),
    input_file: Optional[Path] = typer.Option(None, "--input", "-i", help="JSON input file"),
    benchmark: Optional[str] = typer.Option(None, help="Benchmark name (for generate queue)"),
    difficulty: float = typer.Option(0.5, help="Difficulty level (for generate queue)"),
    count: int = typer.Option(100, help="Problem count (for generate queue)"),
    storage_dir: Optional[Path] = typer.Option(None, help="Task storage directory"),
):
    """Add a task to a queue."""
    import json
    from phfe.orchestrator import QueueType

    queue = _get_task_queue(storage_dir)

    try:
        q = QueueType(queue_name)
    except ValueError:
        console.print(f"[red]Unknown queue: {queue_name}[/red]")
        console.print(f"Available: {', '.join(qt.value for qt in QueueType)}")
        raise typer.Exit(1)

    # Build input data
    if input_file:
        input_data = json.loads(input_file.read_text())
    elif q == QueueType.GENERATE:
        if not benchmark:
            console.print("[red]--benchmark required for generate queue[/red]")
            raise typer.Exit(1)
        input_data = {
            "benchmark": benchmark,
            "difficulty": difficulty,
            "count": count,
        }
    else:
        console.print("[red]Either --input or queue-specific options required[/red]")
        raise typer.Exit(1)

    task_id = queue.enqueue(q, input_data)
    console.print(f"[green]Enqueued task: {task_id}[/green]")
    console.print(json.dumps({"task_id": task_id, "queue": q.value, "input_data": input_data}, indent=2))


@task_app.command("claim")
def task_claim(
    queue_name: str = typer.Argument(..., help="Queue to claim from"),
    worker_type: str = typer.Option(..., "--worker-type", "-w", help="Worker type identifier"),
    worker_id: Optional[str] = typer.Option(None, help="Worker instance ID"),
    debug_mode: bool = typer.Option(False, "--debug", help="Include full context for debugging"),
    storage_dir: Optional[Path] = typer.Option(None, help="Task storage directory"),
):
    """Claim the next available task from a queue."""
    import json
    from phfe.orchestrator import QueueType

    queue = _get_task_queue(storage_dir)

    try:
        q = QueueType(queue_name)
    except ValueError:
        console.print(f"[red]Unknown queue: {queue_name}[/red]")
        raise typer.Exit(1)

    task = queue.claim_task(q, worker_type, worker_id, debug_mode)

    if task is None:
        console.print(f"[yellow]No pending tasks in {queue_name} queue[/yellow]")
        console.print(json.dumps({"task": None, "queue": queue_name}))
        raise typer.Exit(0)

    # Output JSON for tool use parsing
    output = {
        "task_id": task.task_id,
        "queue": task.queue.value,
        "input_data": task.input_data,
        "claimed_by": task.claimed_by,
        "claimed_at": task.claimed_at,
    }

    if debug_mode and task.context_presented:
        output["context_presented"] = task.context_presented

    console.print(json.dumps(output, indent=2))


@task_app.command("submit")
def task_submit(
    task_id: str = typer.Argument(..., help="Task ID to submit result for"),
    result_file: Optional[Path] = typer.Option(None, "--result", "-r", help="JSON result file"),
    concerns_file: Optional[Path] = typer.Option(None, "--concerns", "-c", help="JSON concerns file"),
    storage_dir: Optional[Path] = typer.Option(None, help="Task storage directory"),
):
    """Submit result for a claimed task."""
    import json
    from phfe.orchestrator import WorkerConcern, ConcernLevel

    queue = _get_task_queue(storage_dir)

    # Parse result
    if result_file:
        result = json.loads(result_file.read_text())
    else:
        result = {}

    # Parse concerns
    concerns = None
    if concerns_file:
        concerns_data = json.loads(concerns_file.read_text())
        concerns = [
            WorkerConcern(
                level=ConcernLevel(c["level"]),
                message=c["message"],
                suggestion=c.get("suggestion"),
                context_sample=c.get("context_sample"),
            )
            for c in concerns_data
        ]

    success = queue.submit_result(task_id, result, concerns)

    if success:
        task = queue.get_task(task_id)
        console.print(json.dumps({
            "success": True,
            "task_id": task_id,
            "status": task.status.value if task else "unknown",
        }, indent=2))
    else:
        console.print(json.dumps({
            "success": False,
            "task_id": task_id,
            "error": "Task not found",
        }, indent=2))
        raise typer.Exit(1)


@task_app.command("list")
def task_list(
    queue_name: Optional[str] = typer.Option(None, "--queue", "-q", help="Filter by queue"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    limit: int = typer.Option(20, help="Maximum tasks to show"),
    storage_dir: Optional[Path] = typer.Option(None, help="Task storage directory"),
):
    """List tasks with optional filters."""
    from phfe.orchestrator import QueueType, TaskStatus

    queue = _get_task_queue(storage_dir)

    # Parse filters
    q = None
    if queue_name:
        try:
            q = QueueType(queue_name)
        except ValueError:
            console.print(f"[red]Unknown queue: {queue_name}[/red]")
            raise typer.Exit(1)

    s = None
    if status:
        try:
            s = TaskStatus(status)
        except ValueError:
            console.print(f"[red]Unknown status: {status}[/red]")
            console.print(f"Available: {', '.join(st.value for st in TaskStatus)}")
            raise typer.Exit(1)

    tasks = queue.list_tasks(queue=q, status=s, limit=limit)

    if not tasks:
        console.print("[yellow]No tasks found matching filters[/yellow]")
        return

    table = Table(title=f"Tasks (showing {len(tasks)})")
    table.add_column("ID", style="cyan")
    table.add_column("Queue")
    table.add_column("Status")
    table.add_column("Claimed By")
    table.add_column("Input Preview")

    for t in tasks:
        input_preview = str(t.input_data)[:40] + "..." if len(str(t.input_data)) > 40 else str(t.input_data)
        status_style = {
            TaskStatus.PENDING: "yellow",
            TaskStatus.CLAIMED: "blue",
            TaskStatus.IN_PROGRESS: "blue",
            TaskStatus.COMPLETED: "green",
            TaskStatus.FAILED: "red",
            TaskStatus.NEEDS_REVIEW: "magenta",
        }.get(t.status, "white")

        table.add_row(
            t.task_id,
            t.queue.value,
            f"[{status_style}]{t.status.value}[/{status_style}]",
            t.claimed_by or "-",
            input_preview,
        )

    console.print(table)


@task_app.command("review")
def task_review(
    storage_dir: Optional[Path] = typer.Option(None, help="Task storage directory"),
):
    """Show tasks needing review."""
    import json

    queue = _get_task_queue(storage_dir)
    tasks = queue.get_tasks_needing_review()

    if not tasks:
        console.print("[green]No tasks need review[/green]")
        return

    console.print(f"[yellow]{len(tasks)} tasks need review:[/yellow]\n")

    for t in tasks:
        console.print(f"[cyan]Task ID:[/cyan] {t.task_id}")
        console.print(f"[cyan]Queue:[/cyan] {t.queue.value}")
        console.print(f"[cyan]Claimed by:[/cyan] {t.claimed_by}")

        if t.concerns:
            console.print("[cyan]Concerns:[/cyan]")
            for c in t.concerns:
                level_style = {
                    "info": "blue",
                    "review": "yellow",
                    "retry": "yellow",
                    "error": "red",
                    "escalate": "red bold",
                }.get(c.level.value, "white")
                console.print(f"  [{level_style}][{c.level.value}][/{level_style}] {c.message}")
                if c.suggestion:
                    console.print(f"    Suggestion: {c.suggestion}")

        console.print(f"[cyan]Input:[/cyan]")
        console.print(json.dumps(t.input_data, indent=2))
        console.print("-" * 40)


@task_app.command("retry")
def task_retry(
    task_id: str = typer.Argument(..., help="Task ID to retry"),
    storage_dir: Optional[Path] = typer.Option(None, help="Task storage directory"),
):
    """Re-queue a failed task."""
    queue = _get_task_queue(storage_dir)

    success = queue.retry_task(task_id)

    if success:
        console.print(f"[green]Task {task_id} re-queued as pending[/green]")
    else:
        console.print(f"[red]Cannot retry task {task_id} (not failed or not found)[/red]")
        raise typer.Exit(1)


@task_app.command("concern")
def task_concern(
    task_id: str = typer.Argument(..., help="Task ID to add concern to"),
    level: str = typer.Option(..., "--level", "-l", help="Concern level: info, review, retry, error, escalate"),
    message: str = typer.Option(..., "--message", "-m", help="Concern message"),
    suggestion: Optional[str] = typer.Option(None, "--suggestion", "-s", help="Suggested action"),
    storage_dir: Optional[Path] = typer.Option(None, help="Task storage directory"),
):
    """Add a concern to a task (for workers reporting issues)."""
    from phfe.orchestrator import WorkerConcern, ConcernLevel

    queue = _get_task_queue(storage_dir)

    try:
        concern_level = ConcernLevel(level)
    except ValueError:
        console.print(f"[red]Unknown concern level: {level}[/red]")
        console.print(f"Available: {', '.join(cl.value for cl in ConcernLevel)}")
        raise typer.Exit(1)

    task = queue.get_task(task_id)
    if task is None:
        console.print(f"[red]Task not found: {task_id}[/red]")
        raise typer.Exit(1)

    concern = WorkerConcern(
        level=concern_level,
        message=message,
        suggestion=suggestion,
    )

    # Submit with concern (preserves existing result if any)
    queue.submit_result(task_id, task.result or {}, concerns=[concern])

    console.print(f"[green]Added {level} concern to task {task_id}[/green]")


# === Benchmark Commands ===


@benchmark_app.command("list")
def benchmark_list():
    """List available benchmarks."""
    from phfe.benchmark import BenchmarkType

    table = Table(title="Available Benchmarks")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Type")
    table.add_column("Status")

    for bt in BenchmarkType:
        table.add_row(bt.value, bt.name, "Not loaded")

    console.print(table)


@benchmark_app.command("load")
def benchmark_load(
    benchmark: str = typer.Argument(..., help="Benchmark to load"),
    output_dir: Path = typer.Option("./data", help="Output directory"),
):
    """Load a canonical benchmark."""
    console.print(f"[yellow]Loading benchmark: {benchmark}[/yellow]")
    console.print("[red]Not implemented yet[/red]")


# === ICR Transform Commands ===


@icr_app.command("transform")
def icr_transform(
    benchmark: str = typer.Argument(..., help="Benchmark to transform"),
    tutor_model: str = typer.Option("gpt-4o", help="Tutor model to use"),
    output_dir: Path = typer.Option("./icr_benchmarks", help="Output directory"),
):
    """Transform a benchmark to ICR format."""
    console.print(f"[yellow]Transforming {benchmark} to ICR format[/yellow]")
    console.print(f"Using tutor: {tutor_model}")
    console.print("[red]Not implemented yet[/red]")


# === Corpus Commands ===


@corpus_app.command("build-index")
def corpus_build_index(
    benchmarks: str = typer.Argument(..., help="Comma-separated benchmark list"),
    output_dir: Path = typer.Option("./corpus/canonical_index", help="Output directory"),
):
    """Build canonical benchmark index for contamination checking."""
    console.print(f"[yellow]Building index for: {benchmarks}[/yellow]")
    console.print("[red]Not implemented yet[/red]")


@corpus_app.command("generate")
def corpus_generate(
    benchmark: str = typer.Argument(..., help="Target benchmark"),
    count: int = typer.Option(1000, help="Number of problems to generate"),
    tutor_model: str = typer.Option("deepseek-r1", help="Tutor model"),
    output_dir: Path = typer.Option("./corpus/synthetic", help="Output directory"),
):
    """Generate synthetic problems with answer keys."""
    console.print(f"[yellow]Generating {count} problems for {benchmark}[/yellow]")
    console.print("[red]Not implemented yet[/red]")


@corpus_app.command("stats")
def corpus_stats(corpus_dir: Path = typer.Argument(..., help="Corpus directory")):
    """Show corpus statistics."""
    console.print(f"[yellow]Computing stats for {corpus_dir}[/yellow]")
    console.print("[red]Not implemented yet[/red]")


# === Distillation Commands ===


@distill_app.command("generate")
def distill_generate(
    prompts: Path = typer.Argument(..., help="Prompts file (JSONL)"),
    tutor_model: str = typer.Option("kimi-k2", help="Tutor model"),
    output_dir: Path = typer.Option("./distillation_data", help="Output directory"),
    top_p: float = typer.Option(0.95, help="Top-p for logit storage"),
):
    """Generate distillation data with logits."""
    console.print(f"[yellow]Generating distillation data[/yellow]")
    console.print("[red]Not implemented yet[/red]")


@distill_app.command("train")
def distill_train(
    student_model: Path = typer.Argument(..., help="Student model path"),
    distillation_data: Path = typer.Argument(..., help="Distillation data directory"),
    output_dir: Path = typer.Option("./checkpoints", help="Output directory"),
    steps: int = typer.Option(100000, help="Training steps"),
):
    """Train student with offline distillation."""
    console.print(f"[yellow]Training student model[/yellow]")
    console.print("[red]Not implemented yet[/red]")


# === arXiv Commands ===


@arxiv_app.command("extract")
def arxiv_extract(
    arxiv_id: str = typer.Argument(..., help="arXiv paper ID"),
    output_dir: Path = typer.Option("./arxiv_data", help="Output directory"),
):
    """Extract training data from an arXiv paper."""
    console.print(f"[yellow]Extracting {arxiv_id}[/yellow]")
    console.print("[red]Not implemented yet[/red]")


@arxiv_app.command("compile")
def arxiv_compile(
    tex_file: Path = typer.Argument(..., help="LaTeX file to compile"),
    output_dir: Path = typer.Option("./output", help="Output directory"),
):
    """Compile LaTeX to PDF."""
    console.print(f"[yellow]Compiling {tex_file}[/yellow]")
    console.print("[red]Not implemented yet[/red]")


# === Curriculum Commands ===


@curriculum_app.command("generate")
def curriculum_generate(
    domain: str = typer.Argument(..., help="Domain: math, logic, code"),
    count: int = typer.Option(1000, help="Number of problems"),
    min_difficulty: float = typer.Option(0.1, help="Minimum difficulty"),
    max_difficulty: float = typer.Option(1.0, help="Maximum difficulty"),
    output_dir: Path = typer.Option("./curriculum", help="Output directory"),
):
    """Generate curriculum problems."""
    console.print(f"[yellow]Generating {count} {domain} problems[/yellow]")
    console.print(f"Difficulty range: {min_difficulty} - {max_difficulty}")
    console.print("[red]Not implemented yet[/red]")


# === Eval Commands ===


@eval_app.command("run")
def eval_run(
    generations_file: Path = typer.Argument(..., help="File with generations (one per line)"),
    dimensions: str = typer.Option("all", help="Comma-separated dimensions or 'all'"),
    output: Optional[Path] = typer.Option(None, help="Output JSON file"),
):
    """Evaluate generated texts for language competence."""
    from phfe.language_evals import EvalDimension, evaluate_generations

    console.print(f"[yellow]Evaluating {generations_file}[/yellow]")

    # Read texts
    texts = generations_file.read_text().strip().split("\n")
    console.print(f"Found {len(texts)} texts to evaluate")

    # Parse dimensions
    if dimensions == "all":
        dims = list(EvalDimension)
    else:
        dims = [EvalDimension(d.strip()) for d in dimensions.split(",")]

    # Run evaluation
    scores = evaluate_generations(texts, dims)

    # Display results
    table = Table(title="Language Competence Scores")
    table.add_column("Dimension", style="cyan")
    table.add_column("Score", style="green")

    table.add_row("Syntactic", f"{scores.syntactic:.3f}")
    table.add_row("Lexical", f"{scores.lexical:.3f}")
    table.add_row("Reference", f"{scores.reference:.3f}")
    table.add_row("Discourse", f"{scores.discourse:.3f}")
    table.add_row("Narrative", f"{scores.narrative:.3f}")
    table.add_row("Repetition", f"{scores.repetition:.3f}")
    table.add_row("Aggregate", f"{scores.aggregate:.3f}", style="bold")

    console.print(table)

    if output:
        import json

        output.write_text(
            json.dumps(
                {
                    "syntactic": scores.syntactic,
                    "lexical": scores.lexical,
                    "reference": scores.reference,
                    "discourse": scores.discourse,
                    "narrative": scores.narrative,
                    "repetition": scores.repetition,
                    "aggregate": scores.aggregate,
                },
                indent=2,
            )
        )
        console.print(f"[green]Saved results to {output}[/green]")


if __name__ == "__main__":
    app()
