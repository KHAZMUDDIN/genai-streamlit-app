import streamlit as st
import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import random

from PIL import Image, ImageDraw, ImageFont
import io

from langchain_core.output_parsers import BaseOutputParser
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader


from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import plotly.graph_objects as go


import quiz


# Load environment variables
load_dotenv()

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Learner Pro",
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
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
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

@st.cache_resource
def load_embeddings():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

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
    if "show_quiz" not in st.session_state:
        st.session_state.show_quiz = False

    """Initialize all session state variables safely"""
    if 'test_questions' not in st.session_state:
        st.session_state.test_questions = None
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'test_submitted' not in st.session_state:
        st.session_state.test_submitted = False
    if 'test_results' not in st.session_state:
        st.session_state.test_results = None
    if 'test_history' not in st.session_state:
        st.session_state.test_history = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "config"

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
        # ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
        ["llama-3.3-70b-versatile"],
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

#==============================================================================
#                          Flash Card feature
#==============================================================================

def create_flashcard_image(topic, content, width=800, height=600):
    """
    Create a flashcard as a PIL Image object
    """
    # Create a new image with white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a nice font, fall back to default if not available
    try:
        # You can download and use custom fonts
        title_font = ImageFont.truetype("arial.ttf", 40)
        content_font = ImageFont.truetype("arial.ttf", 28)
        footer_font = ImageFont.truetype("arial.ttf", 16)
    except:
        title_font = ImageFont.load_default()
        content_font = ImageFont.load_default()
        footer_font = ImageFont.load_default()
    
    # Draw decorative border
    border_padding = 20
    draw.rectangle(
        [border_padding, border_padding, width - border_padding, height - border_padding],
        outline='#667eea',
        width=3
    )

    # Draw header background
    header_height = 80
    draw.rectangle([0, 0, width, header_height], fill="#90ea66")
    
    # Draw topic text
    topic_text = f"📚 {topic}"
    # Center the topic text
    bbox = draw.textbbox((0, 0), topic_text, font=title_font)
    topic_width = bbox[2] - bbox[0]
    topic_x = (width - topic_width) // 2
    topic_y = (header_height - 40) // 2
    draw.text((topic_x, topic_y), topic_text, fill='white', font=title_font)
    
    # Draw content with word wrapping
    y_position = header_height + 50
    max_width = width - 80
    words = content.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=content_font)
        line_width = bbox[2] - bbox[0]
        
        if line_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Draw content lines
    for line in lines:
        draw.text((40, y_position), line, fill='#333333', font=content_font)
        y_position += 40
        
        # If content exceeds image height, add "..." and break
        if y_position > height - 100:
            draw.text((40, y_position), "...", fill='#333333', font=content_font)
            break
    
    # Draw footer
    footer_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    bbox = draw.textbbox((0, 0), footer_text, font=footer_font)
    footer_width = bbox[2] - bbox[0]
    draw.text((width - footer_width - 20, height - 30), footer_text, fill='#999999', font=footer_font)
    
    # Draw decorative elements
    # Small dots pattern
    for i in range(10):
        draw.ellipse([width - 30, height - 60 + i*5, width - 25, height - 55 + i*5], fill='#667eea')
    
    return img

def generate_flashcard():
    # Custom output parser to clean the summary
    class SummaryOutputParser(BaseOutputParser):
        def parse(self, text):
            # Clean up the summary text
            text = text.strip()
            # Remove any extra whitespace
            text = ' '.join(text.split())
            return text
    def create_flashcard_summary(content, model="llama-3.3-70b-versatile", temperature=0.3):
        """
        Create a very short summary for flashcard image
        
        Args:
            content (str): The content to summarize
            model (str): Groq model name
            temperature (float): Temperature for LLM (0-1)
        
        Returns:
            str: Summarized content for flashcard
        """
        
        # Create LLM instance
        llm = ChatGroq(
            model=model, 
            temperature=temperature,
        )
        
        # Define the prompt template for summarization
        prompt_template = PromptTemplate(
            template="""You are an expert at creating concise summaries for flashcards. 
    Your task is to summarize the given content into a SHORT, clear, and memorable format suitable for a flashcard image.

    RULES FOR SUMMARY:
    1. Keep it SHORT
    2. Focus on the MOST IMPORTANT key points only
    3. Use simple, easy-to-understand language
    4. Make it visually scannable
    5. Remove any fluff, examples, or redundant information
    6. Ensure the summary captures the CORE concept
    7. If there was any technical term then add it in short

    Content to summarize:
    {content}

    Generate a VERY SHORT summary for a flashcard image:""",
            input_variables=["content"]
        )
        
        # Create and invoke the chain
        chain = prompt_template | llm | SummaryOutputParser()
        result = chain.invoke({"content": content})
        
        return result

    # content = st.session_state.last_result
    content = create_flashcard_summary(st.session_state.last_result,model, temperature)
    
    # if topic and content:
    if st.session_state.last_topic and content:
        st.markdown("---")
        st.markdown("### Generated Flashcard")
        # Image size selection
        st.markdown("### **Image Size**")
        img_size = st.selectbox(
            "",
            ["Small (600x400)", "Medium (800x600)", "Large (1000x700)"],
            label_visibility="collapsed"  # Hide default label
        )
        
        # Map size selection to dimensions
        size_map = {
            "Small (600x400)": (600, 400),
            "Medium (800x600)": (800, 600),
            "Large (1000x700)": (1000, 700)
        }
        width, height = size_map[img_size]
        
        # Generate flashcard based on style
        img = create_flashcard_image(topic, content, width, height)
        
        # Display in columns for better layout
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption=f"Flashcard: {topic}", use_container_width=True)
        
        # Convert PIL Image to bytes for download
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG', quality=95)
        img_byte_arr = img_byte_arr.getvalue()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.download_button(
                label="📥 Download Flashcard as PNG",
                data=img_byte_arr,
                file_name=f"flashcard_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True
            )

def quiz_main():
    st.write("quiz code")
    # ==================== MAIN CONTENT ====================
    st.markdown("AI Test Prep Pro")
    st.markdown("<center><i>Master any topic with AI-generated tests and intelligent analytics</i></center>", unsafe_allow_html=True)
    st.divider()

    # Page Navigation
    if st.session_state.test_submitted:
        current_page = "results"
    elif st.session_state.test_questions:
        current_page = "test"
    else:
        current_page = "config"

    # ==================== CONFIG PAGE ====================
    if current_page == "config":
        st.markdown("### ✏️ Configure Your Test")
        topic=st.session_state.last_topic
        
        # col1, col2 = st.columns([2, 1])
        # 
        # with col1:
            # topic = st.text_input(
            #     "📖 Test Topic",
            #     placeholder="e.g., Machine Learning, Quantum Computing, Biology, etc.",
            #     key="test_topic"
            # )
        # with col2:
        #     difficulty = st.selectbox(
        #         "📊 Difficulty Level",
        #         ["Easy", "Medium", "Hard"],
        #         key="test_difficulty"
        #     )

        difficulty = st.selectbox(
                "📊 Difficulty Level",
                ["Easy", "Medium", "Hard"],
                key="test_difficulty"
            )
        
        st.divider()
        
        st.markdown("### 📋 Question Types")
        col1, col2 = st.columns(2)
        
        question_types = []
        
        with col1:
            if st.checkbox("✅ Multiple Choice (MCQ)", value=True):
                question_types.append("MCQ")
            if st.checkbox("✅ Multiple Select (MSQ)", value=True):
                question_types.append("MSQ")
            if st.checkbox("✅ True/False", value=True):
                question_types.append("True/False")
        
        with col2:
            if st.checkbox("✅ Fill in the Blanks", value=True):
                question_types.append("Fill in the Blanks")
            if st.checkbox("✅ Subjective", value=True):
                question_types.append("Subjective")
        
        st.divider()
        
        num_questions = st.slider(
            "❓ Number of Questions",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            key="num_questions"
        )
        
        temperature = 0.7
        # temperature = st.slider(
        #     "🔥 Creativity Level",
        #     min_value=0.0,
        #     max_value=1.0,
        #     value=0.7,
        #     step=0.1,
        #     help="Lower = More Focused, Higher = More Creative"
        # )
        
        st.divider()
        
        if st.button("🚀 Generate Test", use_container_width=False, type="primary"):
            if not topic or not topic.strip():
                st.error("❌ Please enter a topic!")
            elif not question_types:
                st.error("❌ Please select at least one question type!")
            else:
                with st.spinner("🤖 Generating your test..."):
                    questions = quiz.generate_test_questions(
                        topic=topic,
                        difficulty=difficulty,
                        num_questions=num_questions,
                        question_types=question_types,
                        temperature=temperature
                    )
                    
                    if questions:
                        st.session_state.test_questions = questions
                        st.session_state.test_config = {
                            'topic': topic,
                            'difficulty': difficulty,
                            'num_questions': num_questions,
                            'question_types': question_types
                        }
                        st.session_state.user_answers = {}
                        st.rerun()
                    else:
                        st.error("❌ Failed to generate questions. Please try again.")

    # ==================== TEST PAGE ====================
    elif current_page == "test":
        config = st.session_state.test_config
        
        st.markdown(f"### 📝 {config['topic']} - {config['difficulty']} Level")
        st.caption(f"Total Questions: {config['num_questions']}")
        st.divider()
        
        # Display questions
        for idx, question in enumerate(st.session_state.test_questions):
            with st.container():
                st.markdown(f"**Q{idx + 1}. {question.get('question', '')}**")
                st.caption(f"Type: {question.get('type', 'MCQ')}")
                
                if question.get('type') == 'MCQ' or question.get('type') == 'MSQ':
                    options = question.get('options', [])
                    if question.get('type') == 'MCQ':
                        user_answer = st.radio(
                            "Select answer:",
                            options=options,
                            key=f"q_{idx}",
                            label_visibility="collapsed"
                        )
                        st.session_state.user_answers[idx] = user_answer
                    else:  # MSQ
                        selected = []
                        for option in options:
                            if st.checkbox(option, key=f"msq_{idx}_{option}"):
                                selected.append(option)
                        st.session_state.user_answers[idx] = ", ".join(selected) if selected else ""
                
                elif question.get('type') == 'True/False':
                    answer = st.radio(
                        "Select answer:",
                        options=["True", "False"],
                        key=f"q_{idx}",
                        label_visibility="collapsed"
                    )
                    st.session_state.user_answers[idx] = answer
                
                elif question.get('type') == 'Fill in the Blanks':
                    answer = st.text_input(
                        "Your answer:",
                        key=f"q_{idx}",
                        label_visibility="collapsed"
                    )
                    st.session_state.user_answers[idx] = answer
                
                elif question.get('type') == 'Subjective':
                    answer = st.text_area(
                        "Your answer:",
                        key=f"q_{idx}",
                        label_visibility="collapsed",
                        height=100
                    )
                    st.session_state.user_answers[idx] = answer
                
                st.divider()
        
        # Submit button
        if st.button("✅ Submit Test", use_container_width=True, type="primary"):
            results = quiz.calculate_score(st.session_state.test_questions, st.session_state.user_answers)
            st.session_state.test_results = results
            st.session_state.test_submitted = True
            
            # Add to history
            test_entry = {
                'topic': config['topic'],
                'difficulty': config['difficulty'],
                'correct': results['correct'],
                'incorrect': results['incorrect'],
                'unanswered': results['unanswered'],
                'total': results['total'],
                'percentage': results['percentage'],
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.test_history.append(test_entry)
            
            st.rerun()

    # ==================== RESULTS PAGE ====================
    elif current_page == "results":
        if st.session_state.test_results:
            results = st.session_state.test_results
            config = st.session_state.test_config
            
            st.markdown(f"### 📊 Test Results - {config['topic']}")
            st.divider()
            
            # Score Cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div>✅ Correct</div>
                    <div class="stat-number">{results['correct']}</div>
                    <div>{results['correct']/results['total']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div>❌ Incorrect</div>
                    <div class="stat-number">{results['incorrect']}</div>
                    <div>{results['incorrect']/results['total']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div>⏭️ Unanswered</div>
                    <div class="stat-number">{results['unanswered']}</div>
                    <div>{results['unanswered']/results['total']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="stat-card">
                    <div>📊 Total Score</div>
                    <div class="stat-number">{results['percentage']:.1f}%</div>
                    <div>out of 100</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # AI Feedback
            feedback, suggestions = quiz.generate_ai_feedback(results)
            st.success(feedback)
            
            st.markdown("### 💡 Suggestions for Improvement:")
            for suggestion in suggestions:
                st.info(suggestion)
            
            st.divider()
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie Chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Correct', 'Incorrect', 'Unanswered'],
                    values=[results['correct'], results['incorrect'], results['unanswered']],
                    marker=dict(colors=['#4caf50', '#f44336', '#ff9800'])
                )])
                fig_pie.update_layout(
                    title="Score Distribution",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Bar Chart
                fig_bar = go.Figure(data=[
                    go.Bar(x=['Correct', 'Incorrect', 'Unanswered'],
                        y=[results['correct'], results['incorrect'], results['unanswered']],
                        marker=dict(color=['#4caf50', '#f44336', '#ff9800']))
                ])
                fig_bar.update_layout(
                    title="Question Breakdown",
                    xaxis_title="Status",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.divider()
            
            # Detailed Review
            st.markdown("### 📖 Detailed Review")
            
            for idx, question in enumerate(results['questions']):
                user_answer = results['user_answers'].get(idx, "Not answered")
                correct_answer = question.get('correct_answer', '')
                is_correct = user_answer == correct_answer
                
                status = "✅ Correct" if is_correct else "❌ Incorrect"
                
                with st.expander(f"{status} - Q{idx + 1}: {question.get('question', '')[:50]}..."):
                    st.markdown(f"**Question:** {question.get('question', '')}")
                    st.markdown(f"**Type:** {question.get('type', 'MCQ')}")
                    
                    if 'options' in question:
                        st.markdown("**Options:**")
                        for option in question.get('options', []):
                            st.write(f"  • {option}")
                    
                    st.markdown(f"**Your Answer:** {user_answer}")
                    st.markdown(f"**Correct Answer:** {correct_answer}")
                    st.markdown(f"**Explanation:** {question.get('explanation', 'N/A')}")
            
            st.divider()
            
            # Download PDF and Retry
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pdf = quiz.create_pdf_report(results, config['topic'], config['difficulty'])
                if pdf:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf,
                        file_name=f"{config['topic']}_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("🔄 Retake Test", use_container_width=True):
                    st.session_state.test_questions = None
                    st.session_state.test_submitted = False
                    st.session_state.user_answers = {}
                    st.session_state.test_results = None
                    st.rerun()
            
            with col3:
                if st.button("➕ New Test", use_container_width=True):
                    st.session_state.test_questions = None
                    st.session_state.test_submitted = False
                    st.session_state.user_answers = {}
                    st.session_state.test_results = None
                    st.session_state.current_page = "config"
                    st.rerun()

#==============================================================================
#                          Main Content - original
#==============================================================================
st.markdown(
    "<center><i>**Master any topic with AI-powered personalized explanations**</i></center>", 
    unsafe_allow_html=True
)
col1, col2 = st.columns([4, 1])

with col1:
    st.markdown("### **Enter Topic**")  # Bold and larger using heading
    topic = st.text_input(
        "",
        placeholder="e.g., Quantum Computing, Machine Learning, etc.",
        key="topic_main_input",
        label_visibility="collapsed"  # Hide default label
    )

with col2:
#     if st.button("🎲 Random", use_container_width=True):
#         topic = get_random_topic()
#         st.rerun()
    


    # 1. Create the file uploader widget
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Display a loading spinner while processing
        with st.spinner("Processing PDF..."):
            try:
                # 2. Save the uploaded file to a temporary location
                # PyPDFLoader requires a physical file path string
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                # 3. Feed the temporary file path into PyPDFLoader
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()

                # 4. Clean up the temporary file from disk
                os.remove(temp_file_path)

                # --- Success! Now you can use 'docs' ---
                st.success(f"Successfully loaded {len(docs)} pages!")

                # Optional: Preview the extracted content in the UI
                # st.subheader("Preview Extracted Content")
                # for i, doc in enumerate(docs):
                #     with st.expander(f"Page {i+1}"):
                #         st.write(doc.page_content)

            except Exception as e:
                st.error(f"An error occurred: {e}")
# from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

@st.cache_resource
def create_vectorstore(chunks):

    embeddings = load_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore

if uploaded_file is not None:

    chunks = splitter.split_documents(docs)

    vectorstore = create_vectorstore(chunks)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    st.session_state.retriever = retriever

# Settings Columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### **Explanation Style**")
    way = st.selectbox(
        "",
        ["easy", "technical", "5 year old", "professional", "creative", "comparative"],
        key="style_select",
        label_visibility="collapsed"  # This removes the label space
    )

with col2:
    st.markdown("### **Length**")
    length = st.selectbox(
        "",
        ["very short", "short", "medium", "long", "very long"],
        key="length_select",
        label_visibility="collapsed"  # This removes the label space
    )

with col3:
    include_examples = st.checkbox("Include Examples", value=True)

# Advanced Options
with st.expander("⚡ Advanced Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        include_resources = st.checkbox("📚 Suggest Resources", value=False)
        include_diagrams = st.checkbox("🎨 ASCII Diagrams", value=False)
    
    with col2:
        include_practice = st.checkbox("🎯 Practice Questions", value=False)
        auto_save = st.checkbox("💾 Auto-save Results", value=True)

if st.button("✨ Generate Explanation", use_container_width=False, type="primary"):
    
    if not topic or not topic.strip():
        st.error("❌ Please enter a topic!")
    else:
        with st.spinner("🤖 AI is thinking..."):
            try:
                # Retrieve docs
                retriever = st.session_state.get("retriever")

                if retriever is None:
                    context = "No context"
                else:
                    docs = retriever.invoke(topic)
                    
                    # Combine retrieved chunks
                    context = "\n\n".join(
                        [doc.page_content for doc in docs]
                    )
                prompt_text = """
                You are a helpful AI assistant.

                Use the provided context to answer the question.

                Context:
                {context}

                Question:
                {question}

                Explanation Style:
                {way}

                Length:
                {length}
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
                
                # template = PromptTemplate(
                #     template=prompt_text,
                #     input_variables=[]
                # )
                template = PromptTemplate(
                    template=prompt_text,
                    input_variables=["context", "question", "way", "length"]
                )
                
                chain = template | llm
                # result = chain.invoke({})
                result = chain.invoke({
                    "context": context,
                    "question": topic,
                    "way": way,
                    "length": length
                })
                
                # Store in session state with additional flags
                st.session_state.last_result = result.content
                st.session_state.last_topic = topic
                st.session_state.last_way = way
                st.session_state.last_length = length
                st.session_state.generation_timestamp = datetime.now()
                st.session_state.has_generated = True  # Flag to indicate content exists
                
                # Save to history
                add_to_history(topic, way, length, result.content, model, temperature)
                
                # Display Result
                # st.markdown('<div class="result-container">', unsafe_allow_html=True)
                # st.markdown(f"### 📚 {topic}")
                # st.markdown(f"**Style:** {way} | **Length:** {length}")
                # st.divider()
                # st.markdown(result.content)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Make sure your GROQ_API_KEY is set in .env file")

# Display previously generated content if it exists and no new generation
# Check if we have previously generated content
if st.session_state.get('has_generated', False):
    st.markdown(f"### {st.session_state.get('last_topic', '')}")
    st.markdown(f"**Style:** {st.session_state.get('last_way', '')} | **Length:** {st.session_state.get('last_length', '')}")
    st.divider()
    st.markdown(st.session_state.get('last_result', ''))
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Optional: Show timestamp of when it was generated
    if 'generation_timestamp' in st.session_state:
        st.caption(f"📅 Generated: {st.session_state.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    # if st.button("Generate Summarized Flashcard"):
    #     if st.session_state.get("last_result"):
    #         generate_flashcard()
    #     else:
    #         st.warning("Please generate a result first.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate Summarized Flashcard"):
            if st.session_state.get("last_result"):
                generate_flashcard()
            else:
                st.warning("Please generate a result first.")
    with col2:
        if st.button("Attempt Quiz"):
            st.session_state.show_quiz = True

if st.session_state.get("show_quiz", False):
    quiz_main()