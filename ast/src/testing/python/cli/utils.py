
import asyncio
import random

class DeploymentError(Exception):
    pass

async def fetch_logs(service: str) -> str:
    """Async function to mock fetching logs."""
    await asyncio.sleep(0.1)
    return f"Logs for {service}: [INFO] Started..."

async def deploy_service(config) -> bool:
    """Async deployment simulation."""
    try:
        if config.port == 5000:
             raise DeploymentError("Port 5000 is reserved")
        await asyncio.sleep(0.5)
        return True
    except DeploymentError as e:
        print(f"Deployment failed: {e}")
        return False
    finally:
        print("Cleanup resources...")
