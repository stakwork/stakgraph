
# @ast node: Function "cli"
# @ast node: Function "deploy"
# @ast edge: Calls -> Function "deploy_service" "utils.py"
# @ast edge: Calls -> Function "from_yaml" "config.py"
# @ast node: Function "logs"
# @ast edge: Calls -> Function "fetch_logs" "utils.py"
import click
import asyncio
from .config import AppConfig
from .utils import fetch_logs, deploy_service

@click.group()
def cli():
    """CloudOps CLI Tool."""
    pass

@cli.command()
@click.option('--env', default='dev', help='Environment to deploy to.')
@click.option('--config', default='config.yaml', help='Path to config file.')
def deploy(env: str, config: str):
    """Deploy the service to the specified environment."""
    cfg = AppConfig.from_yaml(config)
    print(f"Deploying to {env} with host {cfg.host}")
    asyncio.run(deploy_service(cfg))

@cli.command()
@click.argument('service_name')
def logs(service_name: str):
    """Fetch logs for a service."""
    logs = asyncio.run(fetch_logs(service_name))
    print(logs)

if __name__ == '__main__':
    cli()
