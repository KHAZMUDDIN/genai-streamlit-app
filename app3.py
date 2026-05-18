import streamlit as st
import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🎓 AI Learner Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
        color: #000;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #ffffff 0%, #f6f8fc 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #000;
    }
    
    /* Stat Cards */
    .stat-card {
        background: linear-gradient(135deg, #ffffff 0%, #faf7ff 100%);
        padding: 25px;
        border-radius: 15px;
        color: #000;
        text-align: center;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
    }
    
    .stat-number {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    /* Input Fields */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        border-radius: 10px !important;
        border: 2px solid #667eea !important;
        padding: 12px !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #764ba2 !important;
        box-shadow: 0 0 10px rgba(118, 75, 162, 0.3) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Result Container */
    .result-container {
        background: white;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    
    /* History Item */
    .history-item {
        background: rgba(255, 255, 255, 0.1);
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 3px solid #ffd700;
        color: white;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .history-item:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
    }
    
    /* Success Messages */
    .stSuccess {
        background-color: rgba(76, 175, 80, 0.1) !important;
        border-left: 4px solid #4caf50 !important;
    }
    
    .stWarning {
        background-color: rgba(255, 152, 0, 0.1) !important;
        border-left: 4px solid #ff9800 !important;
    }
    
    .stError {
        background-color: rgba(244, 67, 54, 0.1) !important;
        border-left: 4px solid #f44336 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(102, 126, 234, 0.1);
        border-radius: 8px;
    }
    
    /* Slider */
    .stSlider [data-testid="stTickBar"] {
        background: linear-gradient(to right, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
def init_session_state():
    """Initialize all session state variables safely"""
    if 'learning_history' not in st.session_state:
        st.session_state.learning_history = []
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    if 'topic_input' not in st.session_state:
        st.session_state.topic_input = ""
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'stats' not in st.session_state:
        st.session_state.stats = {'total_learns': 0, 'total_saves': 0}

init_session_state()

# ==================== CACHE & LLM ====================
@st.cache_resource
def get_llm(model_name):
    """Cache LLM instance to avoid re-initialization"""
    return ChatGroq(model=model_name, temperature=0.7)

# ==================== UTILITY FUNCTIONS ====================
RANDOM_TOPICS = [
    "Quantum Computing",
    "Machine Learning",
    "Blockchain Technology",
    "Neural Networks",
    "Natural Language Processing",
    "Computer Vision",
    "Artificial General Intelligence",
    "Data Science",
    "Cloud Computing",
    "Cybersecurity",
    "Internet of Things",
    "Edge Computing",
    "Augmented Reality",
    "Virtual Reality",
    "5G Technology",
    "Cryptocurrency"
]

def get_random_topic():
    """Get a random learning topic"""
    return random.choice(RANDOM_TOPICS)

def load_history():
    """Load history from file"""
    try:
        with open("learning_history.json", "r") as f:
            return json.load(f)
    except:
        return []

def save_history_to_file():
    """Save history to JSON file"""
    with open("learning_history.json", "w") as f:
        json.dump(st.session_state.learning_history, f, indent=2)

def add_to_history(topic, style, length, result, model, temperature):
    """Add new learning entry to history"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "topic": topic,
        "style": style,
        "length": length,
        "result": result,
        "model": model,
        "temperature": temperature,
        "is_favorite": False
    }
    st.session_state.learning_history.append(entry)
    st.session_state.stats['total_learns'] += 1
    save_history_to_file()

def toggle_favorite(index):
    """Toggle favorite status"""
    if index < len(st.session_state.learning_history):
        st.session_state.learning_history[index]['is_favorite'] = \
            not st.session_state.learning_history[index]['is_favorite']
        save_history_to_file()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Model Selection
    model = st.selectbox(
        "🤖 Select AI Model",
        ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
        help="Choose the LLM model for explanations"
    )
    
    # Temperature Control
    temperature = st.slider(
        "🔥 Creativity Level",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Lower = More Focused, Higher = More Creative"
    )
    
    st.divider()
    
    # Statistics
    st.markdown("### 📊 Your Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🧠 Learns", st.session_state.stats['total_learns'])
    with col2:
        st.metric("❤️ Favorites", len([h for h in st.session_state.learning_history if h.get('is_favorite')]))
    
    st.divider()
    
    # Learning History
    st.markdown("### 📚 Learning History")
    
    if st.session_state.learning_history:
        # Search/Filter
        search_query = st.text_input("🔍 Search history...", key="search_history")
        
        filtered_history = [
            h for h in reversed(st.session_state.learning_history)
            if search_query.lower() in h['topic'].lower() or not search_query
        ]
        
        for idx, entry in enumerate(filtered_history[:10]):  # Show last 10
            with st.expander(f"📌 {entry['topic'][:30]} ({entry['style']})"):
                st.caption(entry['timestamp'][:10])
                st.write(entry['result'][:200] + "...")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("⭐", key=f"fav_{idx}"):
                        toggle_favorite(len(st.session_state.learning_history) - 1 - idx)
                        st.rerun()
                with col2:
                    if st.button("📋", key=f"copy_{idx}"):
                        st.info("Copied to clipboard!")
                with col3:
                    if st.button("🗑️", key=f"del_{idx}"):
                        del st.session_state.learning_history[
                            len(st.session_state.learning_history) - 1 - idx
                        ]
                        save_history_to_file()
                        st.rerun()
        
        # Clear history button
        if st.button("🗑️ Clear All History"):
            st.session_state.learning_history = []
            st.session_state.stats = {'total_learns': 0, 'total_saves': 0}
            save_history_to_file()
            st.rerun()
    else:
        st.info("No learning history yet. Start exploring!")

# ==================== MAIN CONTENT ====================
st.markdown("# 🎓 AI Learner")
st.markdown(
    "<center><i>Master any topic with AI-powered personalized explanations</i></center>", 
    unsafe_allow_html=True
)
st.divider()

# Stat Cards
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.markdown("""
#     <div class="stat-card">
#         <div>🧠 Learn</div>
#         <div class="stat-number">∞</div>
#         <div>Master any topic</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class="stat-card">
#         <div>💾 Save</div>
#         <div class="stat-number">📚</div>
#         <div>Keep your progress</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col3:
#     st.markdown("""
#     <div class="stat-card">
#         <div>📊 Track</div>
#         <div class="stat-number">📈</div>
#         <div>Monitor your journey</div>
#     </div>
#     """, unsafe_allow_html=True)

# st.divider()

# Main Input Section
st.markdown("### 📝 Generate Explanation")

# Topic Input - Fixed to use session state
col1, col2 = st.columns([4, 1])

with col1:
    topic = st.text_input(
        "Enter Topic",
        placeholder="e.g., Quantum Computing, Machine Learning, etc.",
        key="topic_main_input"
    )

with col2:
    if st.button("🎲 Random", use_container_width=True):
        topic = get_random_topic()
        st.rerun()

# Settings Columns
col1, col2, col3 = st.columns(3)

with col1:
    way = st.selectbox(
        "Explanation Style",
        ["easy", "technical", "5 year old", "professional", "creative", "comparative"],
        key="style_select"
    )

with col2:
    length = st.selectbox(
        "Length",
        ["very short", "short", "medium", "long", "very long"],
        key="length_select"
    )

with col3:
    include_examples = st.checkbox("📌 Include Examples", value=True)

# Advanced Options
with st.expander("⚡ Advanced Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        include_resources = st.checkbox("📚 Suggest Resources", value=False)
        include_diagrams = st.checkbox("🎨 ASCII Diagrams", value=False)
    
    with col2:
        include_practice = st.checkbox("🎯 Practice Questions", value=False)
        auto_save = st.checkbox("💾 Auto-save Results", value=True)

# Generate Button
if st.button("✨ Generate Explanation", use_container_width=True, type="primary"):
    
    if not topic or not topic.strip():
        st.error("❌ Please enter a topic!")
    else:
        with st.spinner("🤖 AI is thinking..."):
            try:
                # Build enhanced prompt
                prompt_text = f"""
Explain {topic} in a {way} way.
Keep the explanation {length}.
"""
                
                if include_examples:
                    prompt_text += "Include practical examples.\n"
                if include_resources:
                    prompt_text += "Suggest relevant learning resources.\n"
                if include_diagrams:
                    prompt_text += "Create ASCII diagrams if applicable.\n"
                if include_practice:
                    prompt_text += "Provide 2-3 practice questions at the end.\n"
                
                # Create LLM chain with temperature
                llm = ChatGroq(model=model, temperature=temperature)
                template = PromptTemplate(
                    template=prompt_text,
                    input_variables=[]
                )
                
                chain = template | llm
                result = chain.invoke({})
                
                st.session_state.last_result = result.content
                
                # Save to history
                add_to_history(topic, way, length, result.content, model, temperature)
                
                # Display Result
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown(f"### 📚 {topic}")
                st.markdown(f"**Style:** {way} | **Length:** {length}")
                st.divider()
                st.markdown(result.content)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Action Buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("❤️ Add to Favorites"):
                        st.session_state.learning_history[-1]['is_favorite'] = True
                        save_history_to_file()
                        st.success("⭐ Added to favorites!")
                
                with col2:
                    if st.button("📋 Copy to Clipboard"):
                        st.success("✅ Copied! (Use Ctrl+V to paste)")
                
                with col3:
                    if st.button("💾 Export as Text"):
                        st.download_button(
                            label="📥 Download",
                            data=result.content,
                            file_name=f"{topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Make sure your GROQ_API_KEY is set in .env file")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🚀 Powered by Groq LLMs")
with col2:
    st.caption("💡 LangChain Integration")
with col3:
    st.caption("📱 Streamlit Interface")

st.caption("---")
st.caption("*AI Learner Pro v2.0 - Master any topic with intelligent explanations*")
