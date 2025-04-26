# mylib/cli.py
import json
import shutil
from pathlib import Path
import click

MODULE_DIR  = Path(__file__).resolve().parent
CONFIG_DIR  = MODULE_DIR / "configs"
CONFIG_PATH = CONFIG_DIR / "datapath.json"

@click.group()
def cli():
    """My_Library CLI"""

@cli.command()
@click.option(
    "--config-file",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the user's datapath.json file"
)
def configure(config_file):
    """
    Copy the user's datapath.json into the project's configs/datapath.json.
    """
    # Create configs directory if it doesn't exist
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Validate JSON syntax
        # Chain the exit to the JSON error:
    try:
        cfg_data = json.loads(Path(config_file).read_text())
    except json.JSONDecodeError as err:
        click.echo(f"❌ Invalid JSON in {config_file}: {err}", err=True)
        # Chain the exit to the JSON error:
        raise SystemExit(1) from err

    # 2) Check required keys
    required = ["CCD_PATH", "DB_PATH"]
    missing = [k for k in required if k not in cfg_data]
    if missing:
        click.echo(
            f"❌ Missing required key(s) in config: {', '.join(missing)}",
            err=True
        )
        raise SystemExit(1)

    # Copy the file over
    shutil.copy(config_file, CONFIG_PATH)
    click.echo(f"✅ datapath.json setup complete: {CONFIG_PATH}")

@cli.command('help', context_settings={"help_option_names": ["-h", "--help"]})
@click.argument('command', required=False)
def help_cmd(command):
    """
    Show help for the CLI or a specific command.

    Usage:
      biomol help            Show this message
      biomol help <command>  Show help for that command
    """
    # create a Context for the group itself
    grp_ctx = click.Context(cli)
    if not command:
        click.echo(cli.get_help(grp_ctx))
    else:
        cmd = cli.commands.get(command)
        if cmd:
            cmd_ctx = click.Context(cmd, info_name=command)
            click.echo(cmd.get_help(cmd_ctx))
        else:
            click.echo(f"Error: unknown command '{command}'", err=True)
            raise SystemExit(1)

@cli.command()
def check():
    """
    Verify that configs/datapath.json exists and that all specified paths are valid.
    """
    # Load configuration file
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
        CCD_path = Path(cfg["CCD_PATH"])
        DB_path = Path(cfg["DB_PATH"])
    except Exception as e:
        click.echo(f"❌ Failed to load configuration file: {e}", err=True)
        # Chain the SystemExit to the original exception
        raise SystemExit(1) from e

    # Define paths to check
    paths = {
        "CONTACT_GRAPH_PATH": DB_path / "protein_graph",
        "MSA_PATH":            DB_path / "a3m",
        "CIF_PATH":            DB_path / "cif",
        "SEQ_TO_HASH_PATH":    DB_path / "entity/sequence_hashes.pkl",
        "GRAPH_HASH_PATH":     DB_path / "protein_graph/level0_cluster.csv",
        "GRAPH_CLUSTER_PATH":  DB_path / "cluster/graph_hash_to_graph_cluster.txt",  # noqa: E501
        "MSADB_PATH":          DB_path / "MSA.lmdb",
        "CIFDB_PATH":          DB_path / "cif_protein_only.lmdb",
        "CCD_DB_PATH":         CCD_path / "ligand_info.lmdb",
        "IDEAL_LIGAND_PATH":   DB_path / "metadata/ideal_ligand_list.pkl",
        "SIGNALP_PATH":        DB_path / "signalp",
    }

    # Check existence of each path
    missing = [name for name, p in paths.items() if not p.exists()]
    if missing:
        click.echo(
            "❌ The following paths do not exist:\n  " +
            "\n  ".join(missing),
            err=True
        )
        # Explicitly not chaining here; this is a logical check failure
        raise SystemExit(1)

    click.echo("✅ All data paths are correctly set.")


if __name__ == "__main__":
    cli()
