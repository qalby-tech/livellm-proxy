from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Annotated
from managers.ws import WsManager
from models.ws import WsRequest


ws_router = APIRouter(prefix="/ws", tags=["ws"])


def get_ws_manager(websocket: WebSocket) -> WsManager:
    return websocket.app.state.ws_manager

WsManagerDep = Annotated[WsManager, Depends(get_ws_manager)]

@ws_router.websocket("")
@ws_router.websocket("/") # default route
async def websocket_endpoint(websocket: WebSocket, ws_manager: WsManagerDep):
    """
    Generic WebSocket endpoint over HTTP that accepts any request and routes it to the appropriate handler.
    
    The client sends WsRequest with:
    - action: The action to perform (agent_run, agent_run_stream, audio_speak, etc.)
    - payload: The payload for the action
    
    The server responds with WsResponse containing:
    - status: "success" or "error"
    - action: The action that was performed
    - data: The response data
    - error: Error message if status is "error"
    """
    await websocket.accept()    
    try:
        while True:
            # Receive JSON data from client
            data = await websocket.receive_json()
            request = WsRequest.model_validate(data)
            await ws_manager.handle_request_with_response(websocket, request)
    except WebSocketDisconnect:
        pass