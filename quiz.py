import streamlit as st
import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import random
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import pandas as pd

# Load environment variables
load_dotenv()

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Test Prep Pro",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded"
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
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
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
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.15);
        transition: transform 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.2);
    }
    
    .stat-number {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
        color: #667eea;
    }
    
    /* Input Fields */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 12px !important;
        background: #f8f9fa !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.2) !important;
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
    
    /* Question Container */
    .question-container {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        margin: 15px 0;
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
    
    /* Checkbox */
    .stCheckbox {
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
def init_session_state():
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
def get_llm(model_name, temperature):
    """Cache LLM instance to avoid re-initialization"""
    return ChatGroq(model=model_name, temperature=temperature)

# ==================== UTILITY FUNCTIONS ====================
def generate_test_questions(topic, difficulty, num_questions, question_types, temperature=0.7):
    """Generate test questions using LLM without PromptTemplate"""
    try:
        llm = get_llm("llama-3.3-70b-versatile", temperature)
        
        types_str = ", ".join(question_types)
        
        prompt = f"""Generate {num_questions} exam questions on the topic: "{topic}"

Difficulty Level: {difficulty}
Question Types: {types_str}

IMPORTANT: Return ONLY valid JSON, no other text. Use this exact structure:
{{
    "questions": [
        {{
            "id": 1,
            "type": "MCQ",
            "question": "Question text?",
            "options": ["A) Option1", "B) Option2", "C) Option3", "D) Option4"],
            "correct_answer": "A",
            "explanation": "Explanation here"
        }}
    ]
}}

Create a mix of different question types from: {types_str}
Make sure the questions are {difficulty} level.
Ensure JSON is valid and parseable."""

        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        
        # Parse JSON response
        response_text = response.content
        
        # Extract JSON from response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            questions_data = json.loads(json_str)
            return questions_data.get('questions', [])
        else:
            st.error("Could not parse JSON from response")
            return []
            
    except json.JSONDecodeError as e:
        st.error(f"JSON Parse Error: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return []

def calculate_score(questions, user_answers):
    """Calculate test score and statistics"""
    correct = 0
    incorrect = 0
    unanswered = 0
    
    for i, q in enumerate(questions):
        user_answer = user_answers.get(i, "")
        
        if user_answer == "":
            unanswered += 1
        elif user_answer == q.get('correct_answer', ''):
            correct += 1
        else:
            incorrect += 1
    
    total = len(questions)
    score_percentage = (correct / total * 100) if total > 0 else 0
    
    return {
        'correct': correct,
        'incorrect': incorrect,
        'unanswered': unanswered,
        'total': total,
        'percentage': score_percentage,
        'questions': questions,
        'user_answers': user_answers
    }

def generate_ai_feedback(results):
    """Generate AI feedback based on results"""
    percentage = results['percentage']
    
    if percentage >= 90:
        feedback = "🌟 Excellent! You have mastered this topic. Keep up the great work!"
        suggestions = [
            "✅ Move to advanced topics",
            "✅ Help others understand this topic",
            "✅ Explore related advanced concepts"
        ]
    elif percentage >= 75:
        feedback = "👍 Good job! You have a solid understanding of the topic."
        suggestions = [
            "📌 Review difficult questions",
            "📌 Practice more MCQs",
            "📌 Focus on tricky concepts"
        ]
    elif percentage >= 60:
        feedback = "💭 Average performance. You need more practice on this topic."
        suggestions = [
            "📖 Review fundamental concepts",
            "📖 Watch tutorial videos",
            "📖 Practice subjective questions"
        ]
    else:
        feedback = "🚀 Keep learning! This topic needs more focus."
        suggestions = [
            "📚 Read detailed notes",
            "📚 Start with basic concepts",
            "📚 Take multiple practice tests"
        ]
    
    return feedback, suggestions

def create_pdf_report(results, topic, difficulty):
    """Create a PDF report of the test"""
    try:
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph("📚 AI Test Prep Pro - Report", title_style))
        story.append(Spacer(1, 0.3 * inch))
        
        # Test Info
        info_data = [
            ["Topic", topic],
            ["Difficulty", difficulty],
            ["Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Total Questions", str(results['total'])]
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # Score Summary
        story.append(Paragraph("Score Summary", styles['Heading2']))
        
        score_data = [
            ["Metric", "Count", "Percentage"],
            ["✅ Correct", str(results['correct']), f"{(results['correct']/results['total']*100):.1f}%"],
            ["❌ Incorrect", str(results['incorrect']), f"{(results['incorrect']/results['total']*100):.1f}%"],
            ["⏭️ Unanswered", str(results['unanswered']), f"{(results['unanswered']/results['total']*100):.1f}%"],
            ["📊 Total Score", f"{results['percentage']:.1f}%", ""]
        ]
        
        score_table = Table(score_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(score_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # Questions and Answers
        story.append(Paragraph("Questions & Answers", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))
        
        for idx, question in enumerate(results['questions']):
            user_answer = results['user_answers'].get(idx, "Not answered")
            correct_answer = question.get('correct_answer', '')
            is_correct = user_answer == correct_answer
            
            # Question
            q_text = f"Q{idx + 1}. {question.get('question', '')}"
            story.append(Paragraph(q_text, styles['Heading3']))
            
            # Type
            story.append(Paragraph(f"Type: {question.get('type', 'MCQ')}", styles['Normal']))
            
            # Options
            if 'options' in question:
                for option in question.get('options', []):
                    story.append(Paragraph(f"  • {option}", styles['Normal']))
            
            # User Answer
            answer_color = "green" if is_correct else "red"
            story.append(Paragraph(f"Your Answer: <font color='{answer_color}'>{user_answer}</font>", styles['Normal']))
            story.append(Paragraph(f"Correct Answer: <font color='green'>{correct_answer}</font>", styles['Normal']))
            
            # Explanation
            story.append(Paragraph(f"<b>Explanation:</b> {question.get('explanation', 'N/A')}", styles['Normal']))
            story.append(Spacer(1, 0.2 * inch))
        
        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer
    
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

# ==================== SIDEBAR ====================
# with st.sidebar:
#     st.markdown("### 📊 Test Statistics")
    
#     if st.session_state.test_history:
#         total_tests = len(st.session_state.test_history)
#         avg_score = sum([t['percentage'] for t in st.session_state.test_history]) / total_tests
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("📝 Tests Taken", total_tests)
#         with col2:
#             st.metric("📈 Avg Score", f"{avg_score:.1f}%")
#     else:
#         st.info("📚 No tests taken yet. Start by creating your first test!")
    
#     st.divider()
    
#     # Test History
#     st.markdown("### 📚 Test History")
    
#     if st.session_state.test_history:
#         for idx, test in enumerate(reversed(st.session_state.test_history[-5:])):
#             with st.expander(f"📌 {test['topic'][:20]} - {test['percentage']:.1f}%"):
#                 st.caption(f"Difficulty: {test['difficulty']}")
#                 st.caption(f"Date: {test['date']}")
#                 st.caption(f"Score: {test['correct']}/{test['total']}")
#     else:
#         st.info("No history yet")

# # ==================== MAIN CONTENT ====================
# st.markdown("AI Test Prep Pro")
# st.markdown("<center><i>Master any topic with AI-generated tests and intelligent analytics</i></center>", unsafe_allow_html=True)
# st.divider()

# # Page Navigation
# if st.session_state.test_submitted:
#     current_page = "results"
# elif st.session_state.test_questions:
#     current_page = "test"
# else:
#     current_page = "config"

# # ==================== CONFIG PAGE ====================
# if current_page == "config":
#     st.markdown("### ✏️ Configure Your Test")
    
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         topic = st.text_input(
#             "📖 Test Topic",
#             placeholder="e.g., Machine Learning, Quantum Computing, Biology, etc.",
#             key="test_topic"
#         )
    
#     with col2:
#         difficulty = st.selectbox(
#             "📊 Difficulty Level",
#             ["Easy", "Medium", "Hard"],
#             key="test_difficulty"
#         )
    
#     st.divider()
    
#     st.markdown("### 📋 Question Types")
#     col1, col2 = st.columns(2)
    
#     question_types = []
    
#     with col1:
#         if st.checkbox("✅ Multiple Choice (MCQ)", value=True):
#             question_types.append("MCQ")
#         if st.checkbox("✅ Multiple Select (MSQ)", value=True):
#             question_types.append("MSQ")
#         if st.checkbox("✅ True/False", value=True):
#             question_types.append("True/False")
    
#     with col2:
#         if st.checkbox("✅ Fill in the Blanks", value=True):
#             question_types.append("Fill in the Blanks")
#         if st.checkbox("✅ Subjective", value=True):
#             question_types.append("Subjective")
    
#     st.divider()
    
#     num_questions = st.slider(
#         "❓ Number of Questions",
#         min_value=5,
#         max_value=50,
#         value=10,
#         step=5,
#         key="num_questions"
#     )
    
#     temperature = 0.7
#     # temperature = st.slider(
#     #     "🔥 Creativity Level",
#     #     min_value=0.0,
#     #     max_value=1.0,
#     #     value=0.7,
#     #     step=0.1,
#     #     help="Lower = More Focused, Higher = More Creative"
#     # )
    
#     st.divider()
    
#     if st.button("🚀 Generate Test", use_container_width=False, type="primary"):
#         if not topic or not topic.strip():
#             st.error("❌ Please enter a topic!")
#         elif not question_types:
#             st.error("❌ Please select at least one question type!")
#         else:
#             with st.spinner("🤖 Generating your test..."):
#                 questions = generate_test_questions(
#                     topic=topic,
#                     difficulty=difficulty,
#                     num_questions=num_questions,
#                     question_types=question_types,
#                     temperature=temperature
#                 )
                
#                 if questions:
#                     st.session_state.test_questions = questions
#                     st.session_state.test_config = {
#                         'topic': topic,
#                         'difficulty': difficulty,
#                         'num_questions': num_questions,
#                         'question_types': question_types
#                     }
#                     st.session_state.user_answers = {}
#                     st.rerun()
#                 else:
#                     st.error("❌ Failed to generate questions. Please try again.")

# # ==================== TEST PAGE ====================
# elif current_page == "test":
#     config = st.session_state.test_config
    
#     st.markdown(f"### 📝 {config['topic']} - {config['difficulty']} Level")
#     st.caption(f"Total Questions: {config['num_questions']}")
#     st.divider()
    
#     # Display questions
#     for idx, question in enumerate(st.session_state.test_questions):
#         with st.container():
#             st.markdown(f"**Q{idx + 1}. {question.get('question', '')}**")
#             st.caption(f"Type: {question.get('type', 'MCQ')}")
            
#             if question.get('type') == 'MCQ' or question.get('type') == 'MSQ':
#                 options = question.get('options', [])
#                 if question.get('type') == 'MCQ':
#                     user_answer = st.radio(
#                         "Select answer:",
#                         options=options,
#                         key=f"q_{idx}",
#                         label_visibility="collapsed"
#                     )
#                     st.session_state.user_answers[idx] = user_answer
#                 else:  # MSQ
#                     selected = []
#                     for option in options:
#                         if st.checkbox(option, key=f"msq_{idx}_{option}"):
#                             selected.append(option)
#                     st.session_state.user_answers[idx] = ", ".join(selected) if selected else ""
            
#             elif question.get('type') == 'True/False':
#                 answer = st.radio(
#                     "Select answer:",
#                     options=["True", "False"],
#                     key=f"q_{idx}",
#                     label_visibility="collapsed"
#                 )
#                 st.session_state.user_answers[idx] = answer
            
#             elif question.get('type') == 'Fill in the Blanks':
#                 answer = st.text_input(
#                     "Your answer:",
#                     key=f"q_{idx}",
#                     label_visibility="collapsed"
#                 )
#                 st.session_state.user_answers[idx] = answer
            
#             elif question.get('type') == 'Subjective':
#                 answer = st.text_area(
#                     "Your answer:",
#                     key=f"q_{idx}",
#                     label_visibility="collapsed",
#                     height=100
#                 )
#                 st.session_state.user_answers[idx] = answer
            
#             st.divider()
    
#     # Submit button
#     if st.button("✅ Submit Test", use_container_width=True, type="primary"):
#         results = calculate_score(st.session_state.test_questions, st.session_state.user_answers)
#         st.session_state.test_results = results
#         st.session_state.test_submitted = True
        
#         # Add to history
#         test_entry = {
#             'topic': config['topic'],
#             'difficulty': config['difficulty'],
#             'correct': results['correct'],
#             'incorrect': results['incorrect'],
#             'unanswered': results['unanswered'],
#             'total': results['total'],
#             'percentage': results['percentage'],
#             'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#         st.session_state.test_history.append(test_entry)
        
#         st.rerun()

# # ==================== RESULTS PAGE ====================
# elif current_page == "results":
#     if st.session_state.test_results:
#         results = st.session_state.test_results
#         config = st.session_state.test_config
        
#         st.markdown(f"### 📊 Test Results - {config['topic']}")
#         st.divider()
        
#         # Score Cards
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.markdown(f"""
#             <div class="stat-card">
#                 <div>✅ Correct</div>
#                 <div class="stat-number">{results['correct']}</div>
#                 <div>{results['correct']/results['total']*100:.1f}%</div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         with col2:
#             st.markdown(f"""
#             <div class="stat-card">
#                 <div>❌ Incorrect</div>
#                 <div class="stat-number">{results['incorrect']}</div>
#                 <div>{results['incorrect']/results['total']*100:.1f}%</div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         with col3:
#             st.markdown(f"""
#             <div class="stat-card">
#                 <div>⏭️ Unanswered</div>
#                 <div class="stat-number">{results['unanswered']}</div>
#                 <div>{results['unanswered']/results['total']*100:.1f}%</div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         with col4:
#             st.markdown(f"""
#             <div class="stat-card">
#                 <div>📊 Total Score</div>
#                 <div class="stat-number">{results['percentage']:.1f}%</div>
#                 <div>out of 100</div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.divider()
        
#         # AI Feedback
#         feedback, suggestions = generate_ai_feedback(results)
#         st.success(feedback)
        
#         st.markdown("### 💡 Suggestions for Improvement:")
#         for suggestion in suggestions:
#             st.info(suggestion)
        
#         st.divider()
        
#         # Charts
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Pie Chart
#             fig_pie = go.Figure(data=[go.Pie(
#                 labels=['Correct', 'Incorrect', 'Unanswered'],
#                 values=[results['correct'], results['incorrect'], results['unanswered']],
#                 marker=dict(colors=['#4caf50', '#f44336', '#ff9800'])
#             )])
#             fig_pie.update_layout(
#                 title="Score Distribution",
#                 height=400,
#                 showlegend=True
#             )
#             st.plotly_chart(fig_pie, use_container_width=True)
        
#         with col2:
#             # Bar Chart
#             fig_bar = go.Figure(data=[
#                 go.Bar(x=['Correct', 'Incorrect', 'Unanswered'],
#                        y=[results['correct'], results['incorrect'], results['unanswered']],
#                        marker=dict(color=['#4caf50', '#f44336', '#ff9800']))
#             ])
#             fig_bar.update_layout(
#                 title="Question Breakdown",
#                 xaxis_title="Status",
#                 yaxis_title="Count",
#                 height=400
#             )
#             st.plotly_chart(fig_bar, use_container_width=True)
        
#         st.divider()
        
#         # Detailed Review
#         st.markdown("### 📖 Detailed Review")
        
#         for idx, question in enumerate(results['questions']):
#             user_answer = results['user_answers'].get(idx, "Not answered")
#             correct_answer = question.get('correct_answer', '')
#             is_correct = user_answer == correct_answer
            
#             status = "✅ Correct" if is_correct else "❌ Incorrect"
            
#             with st.expander(f"{status} - Q{idx + 1}: {question.get('question', '')[:50]}..."):
#                 st.markdown(f"**Question:** {question.get('question', '')}")
#                 st.markdown(f"**Type:** {question.get('type', 'MCQ')}")
                
#                 if 'options' in question:
#                     st.markdown("**Options:**")
#                     for option in question.get('options', []):
#                         st.write(f"  • {option}")
                
#                 st.markdown(f"**Your Answer:** {user_answer}")
#                 st.markdown(f"**Correct Answer:** {correct_answer}")
#                 st.markdown(f"**Explanation:** {question.get('explanation', 'N/A')}")
        
#         st.divider()
        
#         # Download PDF and Retry
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             pdf = create_pdf_report(results, config['topic'], config['difficulty'])
#             if pdf:
#                 st.download_button(
#                     label="📥 Download PDF Report",
#                     data=pdf,
#                     file_name=f"{config['topic']}_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
#                     mime="application/pdf",
#                     use_container_width=True
#                 )
        
#         with col2:
#             if st.button("🔄 Retake Test", use_container_width=True):
#                 st.session_state.test_questions = None
#                 st.session_state.test_submitted = False
#                 st.session_state.user_answers = {}
#                 st.session_state.test_results = None
#                 st.rerun()
        
#         with col3:
#             if st.button("➕ New Test", use_container_width=True):
#                 st.session_state.test_questions = None
#                 st.session_state.test_submitted = False
#                 st.session_state.user_answers = {}
#                 st.session_state.test_results = None
#                 st.session_state.current_page = "config"
#                 st.rerun()

# # Footer
# st.divider()
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.caption("🚀 Powered by Groq LLMs")
# with col2:
#     st.caption("💡 LangChain Integration")
# with col3:
#     st.caption("📱 Streamlit Interface")

# st.caption("---")
# st.caption("*AI Test Prep Pro - Master Any Topic with Intelligent Testing*")
