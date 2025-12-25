from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import HTTPException
from typing import Annotated
from managers.config import ConfigManager
from models.common import Settings, SuccessResponse

providers_router = APIRouter(prefix="/providers", tags=["providers"])

def get_config_manager(request: Request) -> ConfigManager:
    return request.app.state.config_manager

ConfigManagerType = Annotated[ConfigManager, Depends(get_config_manager)]

@providers_router.get("/configs")
async def get_configs(config_manager: ConfigManagerType) -> list[Settings]:
    """
    Get all registered provider configurations.

    Returns a list of all provider configurations with masked API keys.
    """
    return list[Settings](config_manager.configs.values())

@providers_router.post("/config", status_code=201)
async def config(request: Settings, config_manager: ConfigManagerType) -> SuccessResponse:
    """
    Add or update a provider configuration.
    
    Args:
        request: Provider configuration including uid, provider type, api_key, and optional base_url
    """
    try:
        await config_manager.add_config(request)
        return SuccessResponse()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@providers_router.delete("/config/{uid}")
async def delete_config(uid: str, config_manager: ConfigManagerType) -> SuccessResponse:
    """
    Delete a provider configuration by uid.
    
    Args:
        uid: The unique identifier of the provider configuration to delete
    """
    try:
        await config_manager.delete_config(uid)
        return SuccessResponse()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

