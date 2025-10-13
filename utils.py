from models.errors import InvalidModelRefError

async def parse_model_ref(model_ref: str) -> tuple[str, str]:
    try:
        provider, model_name = model_ref.split("/", maxsplit=1)
        return provider, model_name
    except ValueError:
        raise InvalidModelRefError(f"Invalid model reference: {model_ref}")

