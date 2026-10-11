import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from lecturelens.config.settings import Settings


import time


def fetch_ollama_models(
    ollama_url: str = "http://localhost:11434",
    max_retries: int = 3,
    retry_delay_seconds: float = 0.8
) -> list[str]:
    url = ollama_url.rstrip("/") + "/api/tags"
    request = urllib.request.Request(url, headers={"User-Agent": "LectureLens"})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=3.0) as response:
                if response.status == 200:
                    payload = json.loads(response.read().decode("utf-8"))
                    models: list[str] = []
                    for item in payload.get("models", []):
                        model_name = item.get("name")
                        if isinstance(model_name, str) and len(model_name.strip()) > 0:
                            models.append(model_name.strip())
                    return models
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay_seconds)
    return []


def filter_gemma4_models(models: list[str]) -> list[str]:
    compatible_models: list[str] = []
    for model in models:
        lower_name = model.lower()
        if "gemma4" in lower_name or "gemma-4" in lower_name:
            compatible_models.append(model)
    return compatible_models


def mask_api_key(key: str) -> str:
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def load_env_api_key(env_path: Path = Path(".env"), var_name: str = "UNOROUTER_API_KEY") -> Optional[str]:
    direct_value = os.environ.get(var_name)
    if direct_value is not None and len(direct_value.strip()) > 0:
        return direct_value.strip()

    lecturelens_value = os.environ.get(f"LECTURELENS_{var_name}")
    if lecturelens_value is not None and len(lecturelens_value.strip()) > 0:
        return lecturelens_value.strip()

    target_prefixes = (f"{var_name}=", f"LECTURELENS_{var_name}=")
    paths_to_check = [env_path]
    if env_path == Path(".env"):
        home_path = Path.home() / ".env"
        if home_path not in paths_to_check:
            paths_to_check.append(home_path)

    for candidate in paths_to_check:
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as file_handle:
                for line in file_handle:
                    stripped = line.strip()
                    for prefix in target_prefixes:
                        if stripped.startswith(prefix):
                            parts = stripped.split("=", 1)
                            if len(parts) == 2:
                                cleaned = parts[1].strip().strip('"').strip("'")
                                if len(cleaned) > 0:
                                    return cleaned
        except Exception:
            continue
    return None


def persist_api_key_to_env(key: str, env_path: Path = Path(".env"), var_name: str = "UNOROUTER_API_KEY") -> None:
    lines: list[str] = []
    found = False
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as file_handle:
                for line in file_handle:
                    if line.strip().startswith(f"{var_name}="):
                        lines.append(f"{var_name}={key}\n")
                        found = True
                    else:
                        lines.append(line)
        except Exception:
            lines = []

    if not found:
        lines.append(f"{var_name}={key}\n")

    try:
        with open(env_path, "w", encoding="utf-8", newline="\n") as file_handle:
            file_handle.writelines(lines)
    except Exception:
        pass


def check_unorouter_health(
    api_key: str,
    base_url: str = "https://api.unorouter.com/v1",
    timeout_seconds: float = 6.0
) -> tuple[bool, str]:
    cleaned_key = api_key.strip()
    if len(cleaned_key) == 0:
        return False, "API key cannot be empty"

    endpoint_url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(
        endpoint_url,
        headers={
            "Authorization": f"Bearer {cleaned_key}",
            "User-Agent": "LectureLens"
        }
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if response.status == 200:
                return True, "API key verified successfully (HTTP 200 OK)"
            return False, f"Unexpected HTTP status {response.status}"
    except urllib.error.HTTPError as error:
        if error.code == 401:
            return False, "Authentication failed (HTTP 401 Unauthorized): Invalid API key"
        if error.code == 403:
            return False, "Access forbidden (HTTP 403 Forbidden): Key not permitted"
        return False, f"HTTP error {error.code}: {error.reason}"
    except urllib.error.URLError as error:
        return False, f"Network connection failed: {error.reason}"
    except Exception as error:
        return False, f"Health check failed: {error}"


def configure_ollama_flow(settings: Settings, console: Console) -> tuple[str, str, str, str]:
    console.print("\n[bold cyan]Checking local Ollama installation...[/bold cyan]")
    models = fetch_ollama_models(settings.ollama_url)

    if len(models) == 0:
        console.print(f"[yellow]Ollama is currently unreachable at {settings.ollama_url}.[/yellow]")
        console.print("[dim]You can still proceed with 'gemma4:e4b' and start Ollama in another terminal.[/dim]")
        choice = Prompt.ask(
            "Select action",
            choices=["default", "custom", "unorouter"],
            default="default"
        )
        if choice == "unorouter":
            return configure_unorouter_flow(console)
        if choice == "custom":
            custom_model = Prompt.ask("Enter custom Ollama model name", default="gemma4:e4b")
            return "gemma", custom_model, "local", "gemma"
        return "gemma", "gemma4:e4b", "local", "gemma"

    compatible = filter_gemma4_models(models)
    if len(compatible) == 0:
        console.print(f"[yellow]Ollama is running, but no Gemma 4 models were detected.[/yellow]")
        console.print(f"[dim]Models installed: {', '.join(models)}[/dim]")
        console.print("[dim]Recommended: Run 'ollama pull gemma4:e4b' in a terminal for local inference.[/dim]")
        choice = Prompt.ask(
            "Select action",
            choices=["pull-default", "custom", "unorouter"],
            default="pull-default"
        )
        if choice == "unorouter":
            return configure_unorouter_flow(console)
        if choice == "custom":
            custom_model = Prompt.ask("Enter Ollama model name to use", default=models[0])
            return "gemma", custom_model, "local", "gemma"
        if choice == "pull-default":
            console.print("[cyan]Running 'ollama pull gemma4:e4b'...[/cyan]")
            try:
                import subprocess
                subprocess.run(["ollama", "pull", "gemma4:e4b"], check=False)
            except Exception as pull_err:
                console.print(f"[yellow]Could not execute ollama pull: {pull_err}[/yellow]")
            return "gemma", "gemma4:e4b", "local", "gemma"
        return "gemma", "gemma4:e4b", "local", "gemma"

    if len(compatible) == 1:
        selected = compatible[0]
        console.print(f"[green]Detected compatible Gemma 4 model: {selected}[/green]")
        return "gemma", selected, "local", "gemma"

    table = Table(title="Available Gemma 4 Models in Ollama")
    table.add_column("Index", style="cyan")
    table.add_column("Model Name", style="green")
    for index, name in enumerate(compatible, start=1):
        table.add_row(str(index), name)
    console.print(table)

    choices = [str(i) for i in range(1, len(compatible) + 1)]
    selection = Prompt.ask("Select model index", choices=choices, default="1")
    selected = compatible[int(selection) - 1]
    return "gemma", selected, "local", "gemma"


def configure_unorouter_flow(console: Console, mode: str = "cloud") -> tuple[str, str, str, str]:
    console.print("\n[bold cyan]Configuring UnoRouter (Multimodal Gateway)...[/bold cyan]")
    existing_key = load_env_api_key()
    api_key: str = ""

    if existing_key is not None and len(existing_key) > 0:
        masked = mask_api_key(existing_key)
        console.print(f"Existing UnoRouter API key found: [green]{masked}[/green]")
        console.print("[dim]Verifying API key health with UnoRouter...[/dim]")
        healthy, health_message = check_unorouter_health(existing_key)
        if healthy:
            console.print(f"[green]UnoRouter API health check: {health_message}[/green]")
            reuse = Prompt.ask("Use this verified key?", choices=["y", "n"], default="y")
            if reuse.lower() == "y":
                api_key = existing_key
        else:
            console.print(f"[yellow]Health check warning: {health_message}[/yellow]")
            reuse = Prompt.ask("Re-enter a new key instead?", choices=["y", "n"], default="y")
            if reuse.lower() == "n":
                api_key = existing_key

    while len(api_key) == 0:
        console.print("[dim]Get your free API key at: https://unorouter.com[/dim]")
        entered_key = Prompt.ask("Paste your UnoRouter API key", password=True).strip()
        if len(entered_key) == 0:
            console.print("[yellow]Key cannot be empty.[/yellow]")
            continue
        console.print("[dim]Verifying API key health with UnoRouter...[/dim]")
        healthy, health_message = check_unorouter_health(entered_key)
        if healthy:
            console.print(f"[green]UnoRouter API health check: {health_message}[/green]")
            api_key = entered_key
            persist_api_key_to_env(api_key)
            break
        else:
            console.print(f"[red]Health check failed: {health_message}[/red]")
            retry = Prompt.ask("Do you want to re-enter?", choices=["y", "n"], default="y")
            if retry.lower() == "n":
                api_key = entered_key
                persist_api_key_to_env(api_key)
                break

    os.environ["UNOROUTER_API_KEY"] = api_key
    os.environ["LECTURELENS_UNOROUTER_API_KEY"] = api_key
    selected_model = "gemma-4-26b:free"
    console.print(f"[green]UnoRouter active with multimodal vision model: {selected_model} (mode: {mode})[/green]")
    return "unorouter", selected_model, mode, "unorouter"


def select_llm_provider(settings: Settings, console: Console) -> tuple[str, str, str, str]:
    console.print(
        Panel(
            "[bold white]Step 1: Choose LLM Provider[/bold white]\n\n"
            "  [1] [bold green]Local Ollama[/bold green] (Offline, RTX GPU/CPU, Gemma 4)\n"
            "  [2] [bold cyan]UnoRouter (Cloud)[/bold cyan] (Pure Cloud, Multimodal Vision gemma-4-26b:free, 262k context)\n"
            "  [3] [bold yellow]Hybrid Mode[/bold yellow] (UnoRouter Cloud with Local Ollama fallback)",
            title="LectureLens AI Backend",
            border_style="blue"
        )
    )
    choice = Prompt.ask("Select provider", choices=["1", "2", "3"], default="1")
    if choice == "2":
        return configure_unorouter_flow(console, mode="cloud")
    if choice == "3":
        return configure_unorouter_flow(console, mode="hybrid")
    return configure_ollama_flow(settings, console)


def render_session_summary_table(options: dict[str, Any]) -> Table:
    table = Table(title="LectureLens Session Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan", width=22)
    table.add_column("Current Value", style="white")

    table.add_row("Session Name", str(options.get("session_id", "")))
    table.add_row(
        "LLM Provider",
        f"{options.get('provider', '')} (model: {options.get('model', '')})"
    )
    table.add_row("LLM Mode", str(options.get("llm_mode", "")))
    table.add_row("Audio Source", str(options.get("audio_source", "")))

    screen_status = "[red]Disabled[/red]" if options.get("no_screen") else f"[green]Enabled[/green] (Monitor {options.get('monitor_index', 1)})"
    table.add_row("Screen Capture", screen_status)

    camera_status = "[red]Disabled[/red]" if options.get("no_camera") else "[green]Enabled[/green]"
    table.add_row("Camera Capture", camera_status)

    course_value = options.get("course") or "None (General)"
    table.add_row("Course Tag", str(course_value))
    return table


def run_interactive_wizard(
    initial_options: dict[str, Any],
    settings: Settings,
    console: Console
) -> Optional[dict[str, Any]]:
    options = dict(initial_options)

    provider, model, llm_mode, primary_provider = select_llm_provider(settings, console)
    options["provider"] = provider
    options["model"] = model
    options["llm_mode"] = llm_mode
    options["primary_provider"] = primary_provider

    while True:
        console.print("\n")
        console.print(render_session_summary_table(options))
        console.print(
            "[bold white]Navigation Options:[/bold white]\n"
            "  [bold green][Enter / 1][/bold green] Start Session\n"
            "  [cyan][2][/cyan] Switch LLM Provider (Ollama / UnoRouter)\n"
            "  [cyan][3][/cyan] Change Audio Source (system / mic / both)\n"
            "  [cyan][4][/cyan] Toggle Screen Capture\n"
            "  [cyan][5][/cyan] Toggle Camera Capture\n"
            "  [cyan][6][/cyan] Change Session Name\n"
            "  [cyan][7][/cyan] Set Course Tag\n"
            "  [red][q][/red] Cancel and Exit\n"
        )

        action = Prompt.ask("Select an option", default="1").strip()

        if action in ("1", "", "s", "start"):
            return options

        if action.lower() in ("q", "quit", "exit"):
            console.print("[yellow]Startup cancelled by user.[/yellow]")
            return None

        if action == "2":
            p, m, lm, pp = select_llm_provider(settings, console)
            options["provider"] = p
            options["model"] = m
            options["llm_mode"] = lm
            options["primary_provider"] = pp

        elif action == "3":
            src = Prompt.ask("Audio source", choices=["system", "mic", "both"], default=options.get("audio_source", "system"))
            options["audio_source"] = src

        elif action == "4":
            options["no_screen"] = not options.get("no_screen", False)

        elif action == "5":
            options["no_camera"] = not options.get("no_camera", False)

        elif action == "6":
            new_name = Prompt.ask("Enter session name", default=options.get("session_id", ""))
            if len(new_name.strip()) > 0:
                options["session_id"] = new_name.strip()

        elif action == "7":
            new_course = Prompt.ask("Enter course name (blank for none)", default=options.get("course") or "")
            options["course"] = new_course.strip() if len(new_course.strip()) > 0 else None
