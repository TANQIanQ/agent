import re
from typing import Dict, Type

from pydantic import BaseModel, field_validator

from vision_agent_tools.shared_types import BaseTool

TOOLS_PATH = "vision_agent_tools.tools"


class ToolRegistryEntry(BaseModel):
    tool_name: str
    class_name: str

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """Ensure tool names are lowercase and separated by underscores."""
        if not re.match(r"^[a-z][a-z0-9_]*$", v):
            raise ValueError(
                f"Tool name '{v}' must be lowercase and separated by underscores."
            )
        return v

    def tool_import(self) -> Type[BaseTool]:
        """Lazy import for a model class."""
        module = __import__(
            f"{TOOLS_PATH}.{self.tool_name}", fromlist=[self.class_name]
        )
        return getattr(module, self.class_name)


TOOL_REGISTRY: Dict[str, ToolRegistryEntry] = {
    "florence2": ToolRegistryEntry(
        tool_name="florence2",
        class_name="Florence2",
    ),
}


def get_tool_class(tool_name: str) -> BaseTool:
    """
    Retrieve a model from the registry based on the model name and task

    Args:
        model_name (str): The name of the model to retrieve

    Returns:
        BaseMLModel: An instance of the requested model

    Raises:
        ValueError: If the model is not registered.
    """

    entry = TOOL_REGISTRY.get(tool_name)

    if not entry:
        raise ValueError(
            f"Tool '{tool_name}' is not registered in the tool registry."
        )

    return entry.tool_import
