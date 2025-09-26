import getpass
import os
import functools
import operator
import glob
import re
import json
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from PIL import Image

import time
import numpy as np
import matplotlib.pyplot as plt

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import tool

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from typing import Sequence, TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from langchain_openai import OpenAIEmbeddings
from dotenv import dotenv_values
from openai import OpenAI

from pathlib import Path
from typing import Literal
import shutil
import requests
from pydantic import BaseModel, Field
from typing import Dict, List
from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel


os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(model="gpt-4.1")


# ## Tools
@tool
def capture_image(source: str, save_dir: str = "./runs/images") -> str:
    """
    Save an image locally and return its absolute file path.
    - If `source` starts with 'http', it will be downloaded.
    - If `source` is a local file path, it will be copied.
    Returns: absolute image path as string.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if source.startswith("http"):
        # download
        filename = source.split("/")[-1] or "image.jpg"
        out_path = Path(save_dir) / filename
        resp = requests.get(source, stream=True, timeout=15)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            shutil.copyfileobj(resp.raw, f)
        return str(out_path.resolve())
    else:
        # local file
        src = Path(source)
        if not src.exists():
            return f'{{"error":"file not found: {source}"}}'
        out_path = Path(save_dir) / src.name
        shutil.copy(src, out_path)
        return str(out_path.resolve())
# ### CLIP
_MODEL = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

_DEF_LABELS = [
    "defocus 0 µm",
    "defocus 8 µm",
    "defocus 16 µm",
    "defocus 24 µm",
    "defocus 32 µm",
    "defocus 40 µm",
]

class ClipDefocusArgs(BaseModel):
    image_source: str = Field(..., description="Image URL or local file path")
    http_timeout: float = Field(15.0, description="HTTP timeout seconds if using URL")

@tool
def clip_defocus_match(args: ClipDefocusArgs) -> Dict:
    """
    Use CLIP to estimate defocus level among fixed labels:
    ['defocus 0 µm','8 µm','16 µm','24 µm','32 µm','40 µm'].
    Returns:
      {
        "top_label": str,
        "scores": [{"label": str, "prob": float}, ...]
      }
    """
    # Load image
    if args.image_source.startswith("http"):
        image = Image.open(requests.get(args.image_source, stream=True, timeout=args.http_timeout).raw).convert("RGB")
    else:
        image = Image.open(args.image_source).convert("RGB")

    # Contextualized prompts (helps CLIP understand the task domain)
    prompts = [
    f"an optical microscope image with out-of-focus blur of {x} micrometers"
    for x in ["0","8","16","24","32","40"]
    ]
     
    inputs = _PROCESSOR(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = _MODEL(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0].tolist()

    scored = [{"label": lab, "prob": float(p)} for lab, p in zip(_DEF_LABELS, probs)]
    scored.sort(key=lambda x: x["prob"], reverse=True)
    return {"top_label": scored[0]["label"], "scores": scored}

@tool
def give_suggestion(clip_result_json: str, confidence_margin: float = 0.15) -> str:
    """
    Provide a simple suggestion based on CLIP result JSON.
    - If top1 - top2 < confidence_margin, suggest collecting more images or labels.
    - Else, recommend the top_label directly.
    Returns a short suggestion string.
    """
    try:
        data = json.loads(clip_result_json)
        if "scores" not in data or not data["scores"]:
            return "Result not valid. Please check image or labels."

        scores = data["scores"]
        if len(scores) == 1:
            return f"Top candidate: {scores[0]['label']} (p={scores[0]['prob']:.3f}). Consider adding more alternatives to verify."

        # top1 and top2
        top1, top2 = scores[0], scores[1]
        if (top1["prob"] - top2["prob"]) < confidence_margin:
            return (
                f"Confidence margin is small (Δ={top1['prob'] - top2['prob']:.3f}). "
                "Consider adding more images, refining labels, or using additional metrics."
            )
        else:
            return f"Recommendation: choose '{top1['label']}' (p={top1['prob']:.3f})."
    except Exception as e:
        return f"Error parsing result: {e}"


# ## Agents

def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant that acquires images, classifies them with CLIP, "
                "and provides actionable suggestions.\n"
                "Available tools: {tool_names}\n"
                "{system_message}"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message) ##used to input {system_message}
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools])) ##used to input {tool_names}
    return prompt | llm.bind_tools(tools)

def agent_node(state, agent, name):
    result = agent.invoke(state)
    # convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.model_dump(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name
    }


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str
    sender: str



from langgraph.prebuilt import ToolNode, tools_condition

def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are an assistant that acquires images, classifies them with CLIP, "
             "and provides actionable suggestions.\n"
             "Available tools: {tool_names}\n"
             "{system_message}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))

    return prompt | llm.bind_tools(tools)

def agent_node(state, agent, name):

    result = agent.invoke(state)

    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.model_dump(exclude={"type", "name"}), name=name)
    return {"messages": [result], "sender": name}

if __name__ == "__main__":

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [capture_image, clip_defocus_match, give_suggestion]

    agent = create_agent(
        llm,
        tools,
        system_message=(
            "Workflow: 1) Use capture_image to ensure a local image path. "
            "2) Use clip_defocus_match to classify defocus levels among the fixed labels. "
            "3) Feed the JSON result to give_suggestion with confidence_margin=0.15, "
            "and output a concise recommendation."
        )
    )

    graph = StateGraph(AgentState)


    def _agent_step(state):
        return agent_node(state, agent, name="assistant")

    graph.add_node("assistant", _agent_step)

    tool_node = ToolNode(tools=tools) 
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "assistant")

    graph.add_conditional_edges(
        "assistant",
        tools_condition, 
        {
            "tools": "tools",
            END: END,
        },
    )

    graph.add_edge("tools", "assistant")

    app = graph.compile()

    init_msg = HumanMessage(content=(
        "Autofocus step: 1) Use capture_image to save a local image from "
        r"C:\Users\Lingchi Deng\Desktop\Script\degraded_z+16um.png "
        "2) Use clip_defocus_match to classify defocus levels among "
        "'defocus 0 µm, defocus 8 µm, defocus 16 µm, defocus 24 µm, defocus 32 µm, defocus 40 µm'; "
        "3) Feed the CLIP result to give_suggestion with confidence_margin=0.15, "
        "and output a concise recommendation for the next focus step."
    ))

    state: AgentState = {
        "messages": [init_msg],
        "next": "",
        "sender": "user",
    }

    final_state = app.invoke(state)

    for m in reversed(final_state["messages"]):
        if isinstance(m, AIMessage):
            print(m.content)
            break
