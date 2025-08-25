#!/usr/bin/env python3
"""
Streamlit interface for the Mexican Revolution RAG Conversational Agent
Provides a modern, interactive web interface with multi-conversation support
"""

from datetime import datetime

import requests
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Mexican Revolution RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-info {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .high-confidence { background-color: #4caf50; color: white; }
    .medium-confidence { background-color: #ff9800; color: white; }
    .low-confidence { background-color: #f44336; color: white; }
    .conversation-item {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        cursor: pointer;
        border: 1px solid #ddd;
    }
    .conversation-item:hover {
        background-color: #f5f5f5;
    }
    .conversation-item.active {
        background-color: #e3f2fd;
        border-color: #2196f3;
    }

</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "conversations" not in st.session_state:
    st.session_state.conversations = {}


def get_confidence_color(confidence):
    """Get color class based on confidence score"""
    if confidence >= 0.7:
        return "high-confidence"
    elif confidence >= 0.4:
        return "medium-confidence"
    else:
        return "low-confidence"


def refresh_conversation_list():
    """Refresh the conversation list from the API"""
    try:
        api_conversations = get_conversations()
        # Update local conversations state
        st.session_state.conversations = {}
        for conv in api_conversations:
            conv_id = conv["conversation_id"]
            st.session_state.conversations[conv_id] = conv
            if conv_id not in st.session_state.messages:
                st.session_state.messages[conv_id] = []
        return True
    except Exception as e:
        st.error(f"Error refreshing conversation list: {str(e)}")
        return False


def get_conversations():
    """Get list of conversations from API"""
    try:
        response = requests.get("http://localhost:8000/conversations", timeout=5)
        if response.status_code == 200:
            return response.json().get("conversations", [])
        else:
            st.error("Failed to fetch conversations")
            return []
    except Exception as e:
        st.error(f"Error fetching conversations: {str(e)}")
        return []


def create_new_conversation():
    """Create a new conversation"""
    conversation_id = f"conv_{datetime.now().timestamp()}"
    st.session_state.current_conversation_id = conversation_id
    st.session_state.messages[conversation_id] = []
    st.session_state.conversations[conversation_id] = {
        "conversation_id": conversation_id,
        "message_count": 0,
        "last_activity": datetime.now().isoformat(),
    }
    return conversation_id


def delete_conversation(conversation_id):
    """Delete a conversation"""
    try:
        response = requests.delete(
            f"http://localhost:8000/conversations/{conversation_id}", timeout=5
        )

        if response.status_code == 200:
            # Remove from local state
            if conversation_id in st.session_state.messages:
                del st.session_state.messages[conversation_id]
            if conversation_id in st.session_state.conversations:
                del st.session_state.conversations[conversation_id]

            # If we deleted the current conversation, create a new one
            if st.session_state.current_conversation_id == conversation_id:
                create_new_conversation()

            st.success("Conversation deleted successfully")

            # Force refresh of conversation list from API
            try:
                api_conversations = get_conversations()
                # Update local conversations state
                st.session_state.conversations = {}
                for conv in api_conversations:
                    conv_id = conv["conversation_id"]
                    st.session_state.conversations[conv_id] = conv
                    if conv_id not in st.session_state.messages:
                        st.session_state.messages[conv_id] = []
            except Exception as e:
                st.warning(f"Could not refresh conversation list: {str(e)}")

            return True
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("detail", "Unknown error")
            except (ValueError, KeyError):
                error_msg = f"HTTP {response.status_code}"

            st.error(f"Failed to delete conversation: {error_msg}")
            return False
    except Exception as e:
        st.error(f"Error deleting conversation: {str(e)}")
        return False


def chat_with_rag(message, conversation_id):
    """Send message to RAG API"""
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={
                "message": message,
                "conversation_id": conversation_id,
            },
            timeout=30,
        )

        if response.status_code == 200:
            return response.json()
        else:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", "Unknown error")
            except (ValueError, KeyError):
                error_detail = f"HTTP {response.status_code}"

            st.error(f"API Error: {error_detail}")
            return None
    except requests.exceptions.Timeout:
        st.error("Connection Error: Request timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(
            "Connection Error: Cannot connect to the API. Please check if the server is running."
        )
        return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None


def summarize_conversation(conversation_id):
    """Generate conversation summary"""
    if not conversation_id:
        st.warning("No conversation to summarize")
        return

    try:
        response = requests.post(
            "http://localhost:8000/summarize",
            json={"conversation_id": conversation_id},
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("summary", "No summary available")
        else:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", "Unknown error")
            except (ValueError, KeyError):
                error_detail = f"HTTP {response.status_code}"

            st.error(f"Summary Error: {error_detail}")
            return None
    except requests.exceptions.Timeout:
        st.error("Summary Error: Request timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(
            "Summary Error: Cannot connect to the API. Please check if the server is running."
        )
        return None
    except Exception as e:
        st.error(f"Summary Error: {str(e)}")
        return None


def classify_conversation(conversation_id):
    """Classify conversation"""
    if not conversation_id:
        st.warning("No conversation to classify")
        return

    try:
        response = requests.post(
            "http://localhost:8000/classify",
            json={"conversation_id": conversation_id},
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            return {
                "category": data.get("category", "unknown"),
                "confidence": data.get("confidence", 0.0),
                "conversation_id": data.get("conversation_id", ""),
            }
        else:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", "Unknown error")
            except (ValueError, KeyError):
                error_detail = f"HTTP {response.status_code}"

            st.error(f"Classification Error: {error_detail}")
            return None
    except requests.exceptions.Timeout:
        st.error("Classification Error: Request timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(
            "Classification Error: Cannot connect to the API. Please check if the server is running."
        )
        return None
    except Exception as e:
        st.error(f"Classification Error: {str(e)}")
        return None


def generate_conversation_title(messages):
    """Generate a short title for the conversation based on the first user message"""
    if not messages:
        return "New Conversation"

    # Find the first user message
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", "")
            # Clean and truncate the title
            content = content.strip()
            if not content:
                continue
            if len(content) > 25:
                title = content[:25] + "..."
            else:
                title = content
            return title

    return "New Conversation"


def get_conversation_title(conversation_id):
    """Get the title for a conversation"""
    messages = st.session_state.messages.get(conversation_id, [])
    title = generate_conversation_title(messages)

    # If we don't have a title yet, try to get it from the conversation data
    if (
        title == "New Conversation"
        and conversation_id in st.session_state.conversations
    ):
        # For conversations that exist but don't have messages loaded yet
        return f"Conversation {conversation_id[:8]}..."

    return title


def get_conversation_display_name(conversation_id, conv_data):
    """Get the display name for a conversation in the sidebar"""
    title = get_conversation_title(conversation_id)
    message_count = conv_data.get("message_count", 0)

    # Truncate title if too long
    if len(title) > 20:
        display_title = title[:20] + "..."
    else:
        display_title = title

    return f"💬 {display_title} ({message_count} msgs)"


# Main header
st.markdown(
    """
<div class="main-header">
    <h1>🤖 Mexican Revolution RAG Agent</h1>
    <p>Ask questions about the Mexican Revolution using AI-powered document retrieval</p>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("🎛️ Controls")

    # Check API health
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success("✅ API Connected")
            st.info(
                f"Active conversations: {health_data.get('active_conversations', 0)}"
            )
        else:
            st.error("❌ API Error")
    except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
        st.error("❌ Cannot connect to API")
        st.info(
            "Make sure the server is running: uvicorn app:app --host 0.0.0.0 --port 8000"
        )

    st.divider()

    # Conversation Management
    st.subheader("💬 Conversations")

    # Create new conversation button
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🆕 New Conversation", use_container_width=True):
            create_new_conversation()
            st.rerun()
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            refresh_conversation_list()
            st.rerun()

    # Get conversations from API
    refresh_conversation_list()

    # Display conversations
    if st.session_state.conversations:
        st.write("**Active Conversations:**")

        for conv_id, conv_data in st.session_state.conversations.items():
            # Determine if this is the active conversation
            is_active = st.session_state.current_conversation_id == conv_id

            # Create a container for each conversation with visual styling
            with st.container():
                # Create the conversation item with styling
                col1, col2 = st.columns([3, 1])

                with col1:
                    # Conversation selection button with custom styling
                    button_text = get_conversation_display_name(conv_id, conv_data)
                    if is_active:
                        # Selected conversation - green background
                        st.markdown(
                            """
                            <div style="background-color: #e8f5e8; border: 2px solid #4caf50; border-radius: 5px; padding: 0.5rem; margin: 0.25rem 0;">
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            button_text,
                            key=f"select_{conv_id}",
                            use_container_width=True,
                            type="secondary",
                        ):
                            st.session_state.current_conversation_id = conv_id
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        # Non-selected conversation - white background
                        st.markdown(
                            """
                            <div style="background-color: white; border: 1px solid #ddd; border-radius: 5px; padding: 0.5rem; margin: 0.25rem 0;">
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            button_text,
                            key=f"select_{conv_id}",
                            use_container_width=True,
                            type="secondary",
                        ):
                            st.session_state.current_conversation_id = conv_id
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    # Delete button
                    if st.button(
                        "🗑️", key=f"delete_{conv_id}", help="Delete conversation"
                    ):
                        if delete_conversation(conv_id):
                            # Force immediate UI update
                            st.rerun()
    else:
        st.info("No conversations yet. Start a new conversation!")

    st.divider()

    # Current conversation tools
    if st.session_state.current_conversation_id:
        st.subheader("🛠️ Tools")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📝 Summarize", use_container_width=True):
                summary = summarize_conversation(
                    st.session_state.current_conversation_id
                )
                if summary:
                    st.info("**Conversation Summary:**")
                    st.write(summary)

        with col2:
            if st.button("🏷️ Classify", use_container_width=True):
                classification = classify_conversation(
                    st.session_state.current_conversation_id
                )
                if classification:
                    st.info("**Conversation Classification:**")
                    category = classification.get("category", "unknown")
                    confidence = classification.get("confidence", 0.0)
                    confidence_percent = confidence * 100

                    # Color coding for categories
                    category_colors = {
                        "historical_figures": "🔵",
                        "historical_events": "🟡",
                        "social_impact": "🟢",
                        "political_aspects": "🟣",
                        "general_information": "⚪",
                        "military_aspects": "🔴",
                    }

                    emoji = category_colors.get(category, "❓")
                    st.write(
                        f"{emoji} **Category:** {category.replace('_', ' ').title()}"
                    )
                    st.write(f"📊 **Confidence:** {confidence_percent:.1f}%")

    st.divider()

    # Example questions
    st.subheader("💡 Example Questions")
    example_questions = [
        "What was the Mexican Revolution?",
        "Who was Porfirio Díaz?",
        "When did the Mexican Revolution start?",
        "Who were the main leaders of the revolution?",
        "What role did Zapata play?",
        "How did the revolution end?",
    ]

    for question in example_questions:
        if st.button(question, key=f"example_{question}"):
            st.session_state.example_question = question
            st.rerun()

# Main chat area
chat_container = st.container()

with chat_container:
    # Ensure we have a current conversation
    if not st.session_state.current_conversation_id:
        create_new_conversation()

    # Display current conversation info
    current_conv_id = st.session_state.current_conversation_id
    conversation_title = get_conversation_title(current_conv_id)
    st.subheader(f"💬 {conversation_title}")

    # Display chat messages for current conversation
    messages = st.session_state.messages.get(current_conv_id, [])

    for message in messages:
        if message["role"] == "user":
            st.markdown(
                f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            confidence_class = get_confidence_color(message["confidence"])
            confidence_percent = message["confidence"] * 100

            st.markdown(
                f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong> {message["content"]}
                <div class="source-info">
                    <span class="confidence-badge {confidence_class}">
                        Confidence: {confidence_percent:.1f}%
                    </span>
                    <br>
                    Sources: {", ".join(message["sources"])}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

# Chat input
st.divider()

# Handle example question
if "example_question" in st.session_state:
    user_input = st.session_state.example_question
    del st.session_state.example_question
else:
    user_input = st.chat_input("Ask about the Mexican Revolution...")

if user_input:
    # Ensure we have a current conversation
    if not st.session_state.current_conversation_id:
        create_new_conversation()

    current_conv_id = st.session_state.current_conversation_id

    # Add user message
    if current_conv_id not in st.session_state.messages:
        st.session_state.messages[current_conv_id] = []

    st.session_state.messages[current_conv_id].append(
        {"role": "user", "content": user_input}
    )

    # Show spinner while processing
    with st.spinner("🤖 Thinking..."):
        # Get response from RAG API
        response = chat_with_rag(user_input, current_conv_id)

        if response:
            # Update conversation ID (in case it changed)
            st.session_state.current_conversation_id = response["conversation_id"]

            # Update local conversation state
            st.session_state.conversations[response["conversation_id"]] = {
                "conversation_id": response["conversation_id"],
                "message_count": len(
                    st.session_state.messages.get(response["conversation_id"], [])
                )
                + 1,
                "last_activity": datetime.now().isoformat(),
            }

            # Add assistant message
            st.session_state.messages[response["conversation_id"]].append(
                {
                    "role": "assistant",
                    "content": response["response"],
                    "confidence": response["confidence"],
                    "sources": response["sources"],
                }
            )
        else:
            st.error("Failed to get response from the API")

    # Rerun to update the display
    st.rerun()

# Footer
st.divider()
st.markdown(
    """
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Powered by RAG (Retrieval-Augmented Generation) Technology</p>
    <p>Built with FastAPI, OpenAI, and Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)
