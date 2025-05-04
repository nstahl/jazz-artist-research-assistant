# Import required packages
from dotenv import load_dotenv
import json
import os
from langchain_openai import ChatOpenAI
import threading
import time
from typing import Optional, Any, Dict
import random
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(temperature=0)

from typing import Optional, Any, Dict
from tavily import TavilyClient

# define search tool
def tavily_custom_search(query: str, include_answer: Optional[str] = None, max_retries: int = 5, base_delay: float = 1.0) -> Dict[Any, Any]:
    """
    Custom Tavily search tool that allows direct configuration of the search with exponential backoff.
    
    Args:
        query: The search query
        include_answer: Type of answer to include
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        
    Returns:
        Dict containing search results
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    # Build search parameters
    search_params = {
        "query": query,
    }
    
    # Only add include_answer if it's provided
    if include_answer is not None:
        search_params["include_answer"] = include_answer

    for attempt in range(max_retries):
        try:
            response = client.search(**search_params)
            return response
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise  # Re-raise the last exception if all retries failed
            
            # Calculate delay with exponential backoff and jitter
            delay = (base_delay * (2 ** attempt)) + (random.random() * 0.1)
            
            # Log the retry attempt
            print(f"Tavily API request failed (attempt {attempt + 1}/{max_retries}). "
                  f"Retrying in {delay:.2f} seconds... Error: {str(e)}")
            
            time.sleep(delay)
            continue


# define state class
from typing import List, Optional
from langchain_core.messages import BaseMessage
from dataclasses import dataclass

@dataclass
class ArtistState:
    artist_name: str
    messages: List[BaseMessage]
    website: Optional[str] = None
    instagram: Optional[str] = None
    youtube_urls: List[str] = None
    biography: Optional[str] = None

    def __str__(self) -> str:
        return f"""
Artist: {self.artist_name}
Website: {self.website or 'Not found'}
Instagram: {self.instagram or 'Not found'}
YouTube URLs: {', '.join(self.youtube_urls) if self.youtube_urls else 'None'}
Biography: {'Found' if self.biography else 'Not found'}
Messages: {len(self.messages)} messages in conversation
"""

    def to_json(self) -> dict:
        return {
            "artist_name": self.artist_name,
            "website": self.website,
            "instagram": self.instagram,
            "youtube_urls": self.youtube_urls if self.youtube_urls else [],
            "biography": self.biography,
            "messages": [
                {
                    "type": "ai" if isinstance(msg, AIMessage) else "human",
                    "content": msg.content
                }
                for msg in self.messages
            ]
        }


# define nodes
from langchain_core.messages import AIMessage

def generate_bio(state: ArtistState):
    result = tavily_custom_search(query=f"{state.artist_name} biography", 
                                  include_answer="advanced")
    new_biography = result["answer"]

    new_message = f"I generated this biography:\n\n{new_biography}"

    # Return a NEW state object
    return ArtistState(
        artist_name=state.artist_name,
        messages=state.messages + [AIMessage(content=new_message)],
        website=state.website,
        instagram=state.instagram,
        youtube_urls=state.youtube_urls,
        biography=new_biography
    )

def find_website(state: ArtistState):
    result = tavily_custom_search(query=f"{state.artist_name} official website -- include url", 
                                  include_answer="basic")
    website_anwer = result["answer"]

    new_message = f"I found this answer to my website search:\n\n{website_anwer}"
    all_messages = state.messages + [AIMessage(content=new_message)]

    import re
    url_pattern = re.compile(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*')
    url_match = url_pattern.search(website_anwer)
    website = url_match.group(0).rstrip(".") if url_match else None
    new_message = f"I extracted this website from the answer:\n\n{website}"
    all_messages = all_messages + [AIMessage(content=new_message)]

    # Return a NEW state object
    return ArtistState(
        artist_name=state.artist_name,
        messages=all_messages,
        website=website,
        instagram=state.instagram,
        youtube_urls=state.youtube_urls,
        biography=state.biography
    )

def find_instagram(state: ArtistState):
    result = tavily_custom_search(query=f"{state.artist_name} official instagram handle -- include instagram handle and url", include_answer="basic")
    instagram_answer = result["answer"]

    new_message = f"I found this answer to my instagram search:\n\n{instagram_answer}"
    all_messages = state.messages + [AIMessage(content=new_message)]

    import re
    url_pattern = re.compile(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*')
    url_match = url_pattern.search(instagram_answer)
    instagram = url_match.group(0).rstrip(".") if url_match else None
    new_message = f"I extracted this instagram from the answer:\n\n{instagram}"
    all_messages = all_messages + [AIMessage(content=new_message)]

    # Return a NEW state object
    return ArtistState(
        artist_name=state.artist_name,
        messages=all_messages,
        website=state.website,
        instagram=instagram,
        youtube_urls=state.youtube_urls,
        biography=state.biography
    )

def find_youtube_videos(state: ArtistState):
    result = tavily_custom_search(query=f"{state.artist_name} performance youtube videos")
    youtube_videos = [search_result["url"] for search_result in result["results"] \
                      if search_result["url"].startswith("https://www.youtube.com") \
                      and not "/videos" in search_result["url"] \
                      and not "/channel/" in search_result["url"] \
                      and not "/user/" in search_result["url"] \
                      and not "/@" in search_result["url"]]

    new_message = f"I found {len(youtube_videos)} youtube videos for {state.artist_name}"
    
    if len(youtube_videos) > 0:
        video_urls_message = ', '.join(youtube_videos)
        new_message = new_message + "\n\n" + video_urls_message

    all_messages = state.messages + [AIMessage(content=new_message)]
        

    # Return a NEW state object
    return ArtistState(
        artist_name=state.artist_name,
        messages=all_messages,
        website=state.website,
        instagram=state.instagram,
        youtube_urls=youtube_videos,
        biography=state.biography
    )


from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# Build graph
builder = StateGraph(ArtistState)
builder.add_node("generate_bio", generate_bio)
builder.add_node("find_website", find_website)
builder.add_node("find_instagram", find_instagram)
builder.add_node("find_youtube_videos", find_youtube_videos)

builder.add_edge(START, "generate_bio")
builder.add_edge("generate_bio", "find_website")
builder.add_edge("find_website", "find_instagram")
builder.add_edge("find_instagram", "find_youtube_videos")
builder.add_edge("find_youtube_videos", END)

graph = builder.compile()

from langchain_core.messages import HumanMessage


concerts_data = []
# Load artist names from jazz_concerts_data.json
with open('scraped_event_details_sorted.json', 'r') as f:
    for line in f:
        concerts_data.append(json.loads(line))
    
# Extract unique artist names from event_title in concerts data
artist_names = list(set(concert['event_title'] for concert in concerts_data if 'event_title' in concert))

artist_states = []

from concurrent.futures import ThreadPoolExecutor
# Create file lock for thread-safe writing
file_lock = threading.Lock()

def process_artist(artist_name):
    # init artist state
    init_artist_state = ArtistState(
        artist_name=artist_name,
        messages=[HumanMessage(content=artist_name)])

    # Get result from graph
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] Fetching results for {artist_name}")
    result = graph.invoke(init_artist_state)
    print(f"[{thread_name}] Got result for {artist_name}")

    # Write result to file immediately with lock
    artist_state = ArtistState(**result)
    with file_lock:
        with open("all_artist_states.json", "a") as f:
            json.dump(artist_state.to_json(), f)
            f.write("\n")
    # Sleep for 3 seconds to avoid Tavily API timeout
    time.sleep(3)
    return artist_state

# Clear file before starting
with open("all_artist_states.json", "w") as f:
    pass

# Process artists concurrently
with ThreadPoolExecutor(max_workers=2) as executor:
    artist_states = list(executor.map(process_artist, artist_names))

print(f"\nWrote all artist states to all_artist_states.json")