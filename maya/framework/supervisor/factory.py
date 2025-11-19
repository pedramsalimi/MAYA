import uuid
import json
import webbrowser
from datetime import datetime
from trustcall import create_extractor
from typing import Optional, Literal, List, Dict, Any, Tuple, TypedDict
from pydantic import BaseModel, Field
import importlib

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs, HumanMessage, SystemMessage, AIMessage, ToolMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

# from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from maya.agents.config import load_agent_specs
from maya.framework.memory import get_postgres_memory   
from dotenv import load_dotenv
from .prompts import TRUSTCALL_INSTRUCTION, MEMORY_UPDATE_INSTRUCTION, SUPERVISOR_SYSTEM_MESSAGE



load_dotenv()
# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model = AzureChatOpenAI(azure_deployment="gpt-4o-mini", temperature=0, api_version="2024-12-01-preview", azure_endpoint="https://mayaagent.openai.azure.com/")
# SCAN_PORTAL_URL = "https://scan.jafarapp.com"
SCAN_PORTAL_URL = "https://plugin-rc.intelliprove.com/?action_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjp7ImVtYWlsIjoiIiwiY3VzdG9tZXIiOiJTRUxGQkFDSy1ERVYiLCJncm91cCI6InVzZXIiLCJtYXhfbWVhc3VyZW1lbnRfY291bnQiOjk5OTksInVzZXJfaWQiOiJiYzc1OTY0MzBjZWE0OTEyYTdmNTI5NzczN2Q4ZmFhYSIsImF1dGgwX3VzZXJfaWQiOm51bGx9LCJtZXRhIjp7fSwiZXhwIjoxNzYzNTgzNDc3fQ.siccXuZLfbw4wdMryddlCovL0PZOya7tsKlhVIHr1hI&language=en&duration=30"

def _load_agents() -> Tuple[List[Any], Dict[str, str]]:
    specs = load_agent_specs()
    agents: List[Any] = []
    descriptions: Dict[str, str] = {}
    for agent_id, spec in specs.items():
        if spec.get("disabled"):
            continue
        module = importlib.import_module(f"maya.agents.{agent_id}.agent")
        agent = module.build(spec)
        name = str(getattr(agent, "name"))
        if name in descriptions:
            raise RuntimeError(f"Duplicate agent name '{name}' detected.")
        agents.append(agent)
        descriptions[name] = (spec.get("description") or "").strip()
    if not agents:
        raise RuntimeError("No enabled agents configured.")
    return agents, descriptions




# Update memory tool
class RouteState(TypedDict):
    """Routing decision for supervisor -> tool nodes."""

    route_type: Literal['user_profile', 'general_memory', 'health_rag', 'scan_portal']

class Memory(BaseModel):
    content: str = Field(description="The main content of the memory. For example: User expressed interest in learning about French.")

# Create the extractor
memory_extractor = create_extractor(
    model,
    tools=[Memory],
    tool_choice="Memory",
    enable_inserts=True,
)

class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list
    )
    interests: list[str] = Field(
        description="Interests that the user has", 
        default_factory=list
    )


# Create the Trustcall extractor for updating the user profile 
profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
)



def build_supervisor():
    def supervisor(state: MessagesState, config: RunnableConfig, store: BaseStore):

        """Load memories from the store and use them to personalize the chatbot's response."""
        
        # Get the user ID from the config
        last_message = state["messages"][-1] if state["messages"] else None
        if isinstance(last_message, ToolMessage) and (last_message.additional_kwargs or {}).get("source") == "health_rag":
            return {"messages": [AIMessage(content=last_message.content)]}

        user_id = config["configurable"]["user_id"]

        # Retrieve profile memory from the store
        namespace = ("profile", user_id)
        memories = store.search(namespace)
        if memories:
            user_profile = memories[0].value
        else:
            user_profile = None
        
        # Retrieve general memory from the store
        namespace = ("memory", user_id)
        memories = store.search(namespace)
        if memories:
            general_memory = "\n".join(json.dumps(mem.value) for mem in memories)
        else:
            general_memory = "No general memories."

        system_msg = SUPERVISOR_SYSTEM_MESSAGE.format(user_profile=user_profile, general_memory=general_memory, time=datetime.now().isoformat())

        # Respond using memory as well as the chat history
        response = model.bind_tools([RouteState], parallel_tool_calls=False).invoke([SystemMessage(content=system_msg)]+state["messages"])

        return {"messages": [response]}

    def update_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):

        """Reflect on the chat history and update the memory collection."""
        
        # Get the user ID from the config
        user_id = config["configurable"]["user_id"]

        # Define the namespace for the memories
        namespace = ("memory", user_id)

        # Retrieve the most recent memories for context
        existing_items = store.search(namespace)

        # Format the existing memories for the Trustcall extractor
        tool_name = "Memory"
        existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                            for existing_item in existing_items]
                            if existing_items
                            else None
                            )

        # Merge the chat history and the instruction
        MEMORY_UPDATE_INSTRUCTION_FORMATTED=MEMORY_UPDATE_INSTRUCTION.format(time=datetime.now().isoformat())
        updated_messages=list(merge_message_runs(messages=[SystemMessage(content=MEMORY_UPDATE_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

        # Invoke the extractor
        result = memory_extractor.invoke({"messages": updated_messages, 
                                            "existing": existing_memories})

        # Save the memories from Trustcall to the store
        for r, rmeta in zip(result["responses"], result["response_metadata"]):
            store.put(namespace,
                    rmeta.get("json_doc_id", str(uuid.uuid4())),
                    r.model_dump(mode="json"),
                )
        tool_calls = state['messages'][-1].tool_calls
        return {"messages": [{"role": "tool", "content": "updated memory", "tool_call_id":tool_calls[0]['id']}]}


    def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):

        """Reflect on the chat history and update the memory collection."""
        
        # Get the user ID from the config
        user_id = config["configurable"]["user_id"]

        # Define the namespace for the memories
        namespace = ("profile", user_id)

        # Retrieve the most recent memories for context
        existing_items = store.search(namespace)

        # Format the existing memories for the Trustcall extractor
        tool_name = "Profile"
        existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                            for existing_item in existing_items]
                            if existing_items
                            else None
                            )

        # Merge the chat history and the instruction
        TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
        updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

        # Invoke the extractor
        result = profile_extractor.invoke({"messages": updated_messages, 
                                            "existing": existing_memories})

        # Save the memories from Trustcall to the store
        for r, rmeta in zip(result["responses"], result["response_metadata"]):
            store.put(namespace,
                    rmeta.get("json_doc_id", str(uuid.uuid4())),
                    r.model_dump(mode="json"),
                )
        tool_calls = state['messages'][-1].tool_calls
        return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id":tool_calls[0]['id']}]}

    def scan_portal_node(
        state: MessagesState,
        config: RunnableConfig,
        store: BaseStore,
    ) -> Dict[str, List[Any]]:
        """Launch the facial scan portal for biomarker capture."""

        messages = state["messages"]
        last = messages[-1] if messages else None
        tool_call_id = (
            last.tool_calls[0]["id"]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None)
            else None
        )

        def send(text: str) -> Dict[str, List[Any]]:
            text = (text or "").strip() or "Scan portal ready."
            if tool_call_id:
                return {
                    "messages": [
                        ToolMessage(
                            content=text,
                            tool_call_id=tool_call_id,
                            name="RouteState",
                        )
                    ]
                }
            return {"messages": [AIMessage(content=text)]}

        url = SCAN_PORTAL_URL
        if isinstance(config, dict):
            configurable = config.get("configurable")
            if isinstance(configurable, dict):
                url = configurable.get("scan_url", url) or url

        try:
            launched = webbrowser.open(url, new=2, autoraise=True)
            error_text = None
        except Exception as exc:
            launched = False
            error_text = str(exc)

        if launched:
            message = (
                f"I've opened {url} in your browser. After starting the scan, keep your face centered and "
                "follow the scan instructions so we can capture the biomarkers. I'll be here when you're done!"
            )
        else:
            message = (
                f"I couldn't automatically open {url}. "
                "Please open it manually and follow the on-screen prompts."
            )
            if error_text:
                message += f"\n\n(Reason: {error_text})"

        return send(message)

    def health_rag_node(
        state: MessagesState,
        config: RunnableConfig,
        store: BaseStore,
    ) -> Dict[str, List[Any]]:
        """Answer medical questions with health_rag + semantic Q→A memory."""

        print("[health_rag_node] called")

        messages = state["messages"]
        agent = agent_by_name.get("health_rag")

        last = messages[-1] if messages else None
        tool_call_id = (
            last.tool_calls[0]["id"]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None)
            else None
        )

        def send(text: str) -> Dict[str, List[Any]]:
            text = (text or "").strip() or "Health specialist completed without response."
            if tool_call_id:
                return {
                    "messages": [
                        ToolMessage(
                            content=text,
                            tool_call_id=tool_call_id,
                            name="RouteState",
                            additional_kwargs={"source": "health_rag"},
                        )
                    ]
                }
            return {"messages": [AIMessage(content=text)]}

        if agent is None:
            return send("Health specialist not available right now.")

        # Latest user question
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        question = human_messages[-1].content.strip() if human_messages else ""
        if not question:
            question = "Please answer the user's latest health question."

        user_id = config["configurable"]["user_id"]
        ns = ("health_memory", user_id)

        # # 1) Semantic search over previous Q→A health memories for THIS user
        # hits = store.search(ns, query=question, limit=1)  # semantic similarity 

        # if hits:
        #     value = hits[0].value or {}
        #     answer = value.get("answer")
        #     if isinstance(answer, str) and answer.strip():
        #         print("[health_rag_node] reusing from memory")
        #         return send(
        #             "As we discussed earlier, here is the explanation again:\n\n"
        #             + answer
        #         )
        

        # 2) No good hit → call health_rag agent (with profile-aware instructions)

        profile_docs = store.search(("profile", user_id))
        agent_messages: List[Any] = []

        if profile_docs:
            profile = profile_docs[0].value
            agent_messages.append(
                SystemMessage(
                    content=(
                        "User background/profile (adapt explanation depth and tone):\n"
                        f"{json.dumps(profile, ensure_ascii=False)}\n\n"
                        "- If the user is NOT from a medical/scientific background, use simple language, "
                        "short sentences, and concrete examples.\n"
                        "- If the user DOES have a medical/scientific background, add a short "
                        "'Technical details' section.\n"
                        "- In ALL cases, base claims strictly on PubMed evidence from the "
                        "`pubmed_health_rag` tool and ALWAYS include inline numeric citations "
                        "[1], [2] and a References section using ONLY the tool-provided citations.\n"
                    )
                )
            )

        agent_messages.append(HumanMessage(content=question))

        result = agent.invoke({"messages": agent_messages}, config=config)

        if isinstance(result, dict) and isinstance(result.get("messages"), list) and result["messages"]:
            answer = result["messages"][-1].content
        else:
            answer = getattr(result, "content", str(result))

        answer = (answer or "").strip() or "Health specialist completed without response."
        print(f"[health_rag_node answer]: {answer}")

        summary_prompt = (
        "Summarise the following medical explanation into maximum two sentences"
        "in clear, empathetic language suitable for user background below:"
        f"{json.dumps(profile, ensure_ascii=False)}. "
        "Do not include citations, references, or markdown:\n\n"
        f"{answer}"
    )

        summary = model.invoke(summary_prompt).content.strip()
        if not summary:
            summary = answer
        # 3) Store new Q→A with "question" field indexed for future semantic search
        store.put(
            ns,
            key=str(uuid.uuid4()),
            value={
                "question": question,
                "answer": answer,
                "created_at": datetime.utcnow().isoformat(),
            },
            index=["question"],  # tell the store which field to embed for this doc 
        )
        
        return send(summary)

    def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_profile", "update_memory", "health_rag", "scan_portal"]:

        """
        Use this tool to decide where to route the user’s request:
        - 'health_rag' for ANY medical/health-related question
        - 'user_profile' for stable personal facts
        - 'general_memory' for experiences / events / feelings
        - 'scan_portal' when the user asks to run the facial scan.
        """

        message = state['messages'][-1]
        if len(message.tool_calls) == 0:
            return END
        else:
            tool_call = message.tool_calls[0]
            if tool_call['args']['route_type'] == "user_profile":
                return "update_profile"
            elif tool_call['args']['route_type'] == "general_memory":
                return "update_memory"
            elif tool_call['args']['route_type'] == "health_rag":
                return "health_rag"
            elif tool_call['args']['route_type'] == "scan_portal":
                return "scan_portal"
            else:
                raise ValueError
            



    store, checkpointer = get_postgres_memory()
    agents, _ = _load_agents()
    agent_by_name = {a.name: a for a in agents}
    # Create the graph + all nodes
    builder = StateGraph(MessagesState)

    # Define the flow of the memory extraction process
    builder.add_node(supervisor)
    builder.add_node(update_profile)
    builder.add_node(update_memory)
    builder.add_node("scan_portal", scan_portal_node)

    for agent in agents:
        if agent.name != "health_rag":
            builder.add_node(agent.name, agent)

    builder.add_node("health_rag", health_rag_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route_message)
    builder.add_edge("update_profile", "supervisor")
    builder.add_edge("update_memory", "supervisor")
    builder.add_edge("scan_portal", "supervisor")
    builder.add_edge("health_rag", "supervisor")


    # We compile the graph with the checkpointer and store
    graph = builder.compile(checkpointer=checkpointer, store=store)
    return graph
