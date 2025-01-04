# AI-Agents - Beginners Guide


From Generative AI to AI Agents, what are AI Agents. Evolution of generative AI
![AI Agents Overview and Evolution](https://skolo-online.ams3.cdn.digitaloceanspaces.com/ai-agents/overview.png)


## Chapter 1: PydanticAI 

### Installation 

Before you get started - check for the python version that you are on. PydanticAI requires python time 3.9+

```
python --version
```
If the version you see is less than 3.9 then upgrade you python.

Create a virtualenv and install 
```
virtualenv skoloenv
source skoloenv/bin/activate
pip install pydantic-ai
```

### Pydantic AI Basics

Agent class is the main class within PydanticAI, you can read more about it [here in the documentation](https://ai.pydantic.dev/api/agent/).

Available Models at time of print:
```
KnownModelName = Literal[
    "openai:gpt-4o",
    "openai:gpt-4o-mini",
    "openai:gpt-4-turbo",
    "openai:gpt-4",
    "openai:o1-preview",
    "openai:o1-mini",
    "openai:o1",
    "openai:gpt-3.5-turbo",
    "groq:llama-3.3-70b-versatile",
    "groq:llama-3.1-70b-versatile",
    "groq:llama3-groq-70b-8192-tool-use-preview",
    "groq:llama3-groq-8b-8192-tool-use-preview",
    "groq:llama-3.1-70b-specdec",
    "groq:llama-3.1-8b-instant",
    "groq:llama-3.2-1b-preview",
    "groq:llama-3.2-3b-preview",
    "groq:llama-3.2-11b-vision-preview",
    "groq:llama-3.2-90b-vision-preview",
    "groq:llama3-70b-8192",
    "groq:llama3-8b-8192",
    "groq:mixtral-8x7b-32768",
    "groq:gemma2-9b-it",
    "groq:gemma-7b-it",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash-exp",
    "vertexai:gemini-1.5-flash",
    "vertexai:gemini-1.5-pro",
    "mistral:mistral-small-latest",
    "mistral:mistral-large-latest",
    "mistral:codestral-latest",
    "mistral:mistral-moderation-latest",
    "ollama:codellama",
    "ollama:gemma",
    "ollama:gemma2",
    "ollama:llama3",
    "ollama:llama3.1",
    "ollama:llama3.2",
    "ollama:llama3.2-vision",
    "ollama:llama3.3",
    "ollama:mistral",
    "ollama:mistral-nemo",
    "ollama:mixtral",
    "ollama:phi3",
    "ollama:qwq",
    "ollama:qwen",
    "ollama:qwen2",
    "ollama:qwen2.5",
    "ollama:starcoder2",
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-opus-latest",
    "test",
]
```
We will use OpenAI models, so start be setting the API key as an environment variable:

```sh
export OPENAI_API_KEY='your-api-key'
```

Method 2: Manual (Not Recommended)

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# Define the Model First with the API Key
model = OpenAIModel('gpt-4o', api_key='add-your-api-key-here')
# Call the agent with the model defined above
agent = Agent(model)


result = agent.run_sync("What is Bitcoin")
print(result.data)
```

## PydanticAI with Tools
The reason why pydanticAI is so powerful is --- structured data. You can utilise Pydantic to create structured outputs that you can use with tools.
We will build an example "Note Manager". The note manager is an assistant that allows you to create, list and retrieve notes. This is a perfect example to demostrate how powerful PydanticAI is.
For this we will use a PostgreSQL database, here is the code for the database:

You will need to install the following:
```sh
pip install psycopg2
pip install asyncpg
```

```python

import psycopg2
DB_DSN = "database-connection-string"



def create_notes_table():
    """
    Creates a table named 'notes' with columns 'title' (up to 200 characters) and 'text' (unlimited length).

    """

    create_table_query = """
    CREATE TABLE IF NOT EXISTS notes (
        id SERIAL PRIMARY KEY,
        title VARCHAR(200) UNIQUE NOT NULL,
        text TEXT NOT NULL
    );
    """
    try:
        # Connect to the database
        connection = psycopg2.connect(DB_DSN)
        cursor = connection.cursor()
        
        # Execute the table creation query
        cursor.execute(create_table_query)
        connection.commit()
        
        print("Table 'notes' created successfully (if it didn't already exist).")
    
    except psycopg2.Error as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Close the connection
        if connection:
            cursor.close()
            connection.close()


def check_table_exists(table_name: str) -> bool:
    """
    Checks if a table exists in the PostgreSQL database.

    Args:
        table_name (str): The name of the table to check.

    Returns:
        bool: True if the table exists, False otherwise.
    """

    query = """
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name = %s
    );
    """
    try:
        connection = psycopg2.connect(DB_DSN)
        cursor = connection.cursor()
        cursor.execute(query, (table_name,))
        exists = cursor.fetchone()[0]
        return exists
    except psycopg2.Error as e:
        print(f"An error occurred: {e}")
        return False
    finally:
        if connection:
            cursor.close()
            connection.close()


"""
Asynchronous database connection class for interacting with the 'notes' table in your PostgreSQL database.

This class provides methods to add a note, retrieve a note by title,
and list all note titles.
"""

import asyncpg
from typing import Optional, List


class DatabaseConn:
    def __init__(self):
        """
        Initializes the DatabaseConn with the Data Source Name (DSN).
        """
        self.dsn = DB_DSN

    async def _connect(self):
        """
        Establishes an asynchronous connection to the PostgreSQL database.

        Returns:
            asyncpg.Connection: An active database connection.
        """
        return await asyncpg.connect(self.dsn)

    async def add_note(self, title: str, text: str) -> bool:
        """
        Adds a new note to the 'notes' table.

        Args:
            title (str): The title of the note.
            text (str): The content of the note.

        Returns:
            bool: True if the note was added successfully, False otherwise.
        """
        query = """
        INSERT INTO notes (title, text)
        VALUES ($1, $2)
        ON CONFLICT (title) DO NOTHING;
        """
        conn = await self._connect()
        try:
            result = await conn.execute(query, title, text)
            return result == "INSERT 0 1"  # Returns True if one row was inserted
        finally:
            await conn.close()

    async def get_note_by_title(self, title: str) -> Optional[dict]:
        """
        Retrieves a note's content by its title.

        Args:
            title (str): The title of the note to retrieve.

        Returns:
            Optional[dict]: A dictionary containing the note's title and text if found, None otherwise.
        """
        query = "SELECT title, text FROM notes WHERE title = $1;"
        conn = await self._connect()
        try:
            result = await conn.fetchrow(query, title)
            if result:
                return {"title": result["title"], "text": result["text"]}
            return None
        finally:
            await conn.close()

    async def list_all_titles(self) -> List[str]:
        """
        Lists all note titles in the 'notes' table.

        Returns:
            List[str]: A list of all note titles.
        """
        query = "SELECT title FROM notes ORDER BY title;"
        conn = await self._connect()
        try:
            results = await conn.fetch(query)
            return [row["title"] for row in results]
        finally:
            await conn.close()



```

 Note: Watch the full video to see how to get a DB connetion string

 Check that the DB connection is working before continuing to the next step:

 ```python
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, Tool
from typing import Optional, List
from database import DatabaseConn

from pydantic_ai.models.openai import OpenAIModel

OPENAI_API_KEY="enter-your-openai-api-key-here"

# Dataclass for structured output from Intent Extraction Agent
@dataclass
class NoteIntent:
    action: str
    title: Optional[str] = None
    text: Optional[str] = None

# Dataclass for dependencies
@dataclass
class NoteDependencies:
    db: DatabaseConn

# Define the response structure
class NoteResponse(BaseModel):
    message: str
    note: Optional[dict] = None
    titles: Optional[List[str]] = None


# Intent Extraction Agent
intent_model = OpenAIModel('gpt-4o-mini', api_key=OPENAI_API_KEY)
intent_agent = Agent(
    intent_model,
    result_type=NoteIntent,
    system_prompt=(
        "You are an intent extraction assistant. Your job is to analyze user inputs, "
        "determine the user's intent (e.g., create, retrieve, list), and extract relevant "
        "information such as title and text. Output the structured intent in the following format:\n\n"
        "{'action': '...', 'title': '...', 'text': '...'}\n\n"
        "Always use MD format for the text field.",
        "Add supporting information to help the user understand the note and produce well formatted notes.",
        "For example, if the user says 'Please note the following: Meeting tomorrow at 10 AM', "
        "the output should be:\n{'action': 'create', 'title': 'Meeting', 'text': '## Meeting Note \n\n Please note a meeting on the 4th of January at 10 AM.'}."
    )
)

# Action Handling Agent
action_model = OpenAIModel('gpt-4o-mini', api_key=OPENAI_API_KEY)
action_agent = Agent(
    action_model,
    deps_type=NoteDependencies,
    result_type=NoteResponse,
    system_prompt=(
        "You are a note management assistant. Based on the user's intent (action, title, text), "
        "perform the appropriate action: create a note, retrieve a note, or list all notes."
    )
)

# Define tools for Action Handling Agent
@action_agent.tool
async def create_note_tool(ctx: RunContext[NoteDependencies], title: str, text: str) -> NoteResponse:
    db = ctx.deps.db
    success = await db.add_note(title, text) 
    return NoteResponse(message="CREATED:SUCCESS") if success else NoteResponse(message="CREATED:FAILED")

@action_agent.tool
async def retrieve_note_tool(ctx: RunContext[NoteDependencies], title: str) -> NoteResponse:
    db = ctx.deps.db
    note = await db.get_note_by_title(title)
    return NoteResponse(message="GET:SUCCESS", note=note) if note else NoteResponse(message="GET:FAILED")

@action_agent.tool
async def list_notes_tool(ctx: RunContext[NoteDependencies]) -> NoteResponse:
    db = ctx.deps.db
    titles = await db.list_all_titles()
    return NoteResponse(message="LIST:SUCCESS", titles=titles)

# Main function to handle user input with both agents
async def handle_user_query(user_input: str, deps: NoteDependencies) -> NoteResponse:
    intent = await intent_agent.run(user_input)
    print(intent.data)
    
    if intent.data.action == "create":
        query = f"Create a note with the title '{intent.data.title}' and the text '{intent.data.text}'."
        response = await action_agent.run(query, deps=deps)
        return response.data
    elif intent.data.action == "retrieve":
        query = f"Retrieve the note with the title '{intent.data.title}'."
        response = await action_agent.run(query, deps=deps)
        return response.data
    elif intent.data.action == "list":
        query = "List all notes."
        response = await action_agent.run(query, deps=deps)
        return response.data
    else:
        return NoteResponse(message="Invalid action. Please try again.")

# Example usage 
async def ask(query: str):
    db_conn = DatabaseConn()
    note_deps = NoteDependencies(db=db_conn)
    response = await handle_user_query(query, note_deps)
    return response

# from main import *
# query = ""
# import asyncio
# asyncio.run(ask(query))

```

### Streamlit App Interface
Final step is to build a user friendly interface to interact with our agent:

Install streamlit:
```sh
pip install streamlit
```

```python
# pip install streamlit
# streamlit run app.py

import asyncio
import streamlit as st
from main import ask # The Ask Function is in the main.py file

# Set up Streamlit page
st.set_page_config(page_title="Note Management Agent", layout="centered")

st.title("Note Management System")
st.write("Ask the agent to create, retrieve, or list notes.")

# User input
user_input = st.text_area("Enter your query:", placeholder="e.g., Create a note titled 'Meeting' about tomorrow's meeting.")

if st.button("Submit"):
    if not user_input.strip():
        st.error("Please enter a valid query.")
    else:
        # Run the `ask` function asynchronously
        with st.spinner("Processing..."):
            try:
                response = asyncio.run(ask(user_input)) 
                
                # Handle the response based on the presence of None in `note` and `titles`
                if response.note is None and response.titles is None:
                    st.success(response.message)
                elif response.note is not None and response.titles is None:
                    note = response.note
                    st.success(response.message)
                    st.subheader(f"Note: {note.get('title', 'Untitled')}")
                    st.write(note.get('text', "No content available."))
                elif response.note is None and response.titles is not None:
                    titles = response.titles
                    st.success(response.message)
                    st.subheader("List of Notes:")
                    if titles:
                        for title in titles:
                            st.write(f"- {title}")
                    else:
                        st.info("No notes available.")
                else:
                    st.error("Unexpected response format. Please check your query.")
            except Exception as e:
                st.error(f"An error occurred: {e}")


```


RUN Streamlit, assuming the file is app.py

```sh
streamlit run app.py
```



