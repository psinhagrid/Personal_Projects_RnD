import json
import re
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
import os

class AnalysisResult(BaseModel):
    has_ambiguity: bool = Field(description="Whether the question has ambiguity or context is insufficient")
    confidence_level: str = Field(description="HIGH, MEDIUM, or LOW confidence in answering")
    reasoning: str = Field(description="Explanation of the analysis")
    answer: Optional[str] = Field(description="Direct answer if no ambiguity", default=None)
    clarifying_questions: Optional[List[str]] = Field(description="List of clarifying questions", default=None)
    contradictions: Optional[List[str]] = Field(description="List of contradictions found", default=None)

class OpenAIQASystem:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the QA system with ChatOpenAI
        
        Args:
            api_key: OpenAI API key (optional, can use OPENAI_API_KEY env var)
            model_name: OpenAI model to use (default: gpt-3.5-turbo)
        """
        # Use ChatOpenAI with your specific configuration
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            max_tokens=1024,
            timeout=60.0,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url='https://genai-api-visa.com/v1'
        )
        
        self.model_name = model_name

    def _call_openai(self, prompt: str, system_message: str = None) -> str:
        """Make a call to ChatOpenAI"""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error calling ChatOpenAI: {str(e)}"

    def analyze_intent(self, question: str) -> str:
        """Analyze the intent of the question"""
        prompt = f"""
Analyze what specific information the user is asking for:

Question: {question}

Identify:
1. The main subject/entity being asked about
2. The specific attribute or information requested
3. Any qualifiers or conditions in the question

Provide a brief analysis in 2-3 sentences.
"""
        return self._call_openai(prompt)

    def detect_ambiguity(self, question: str, context: str, intent_analysis: str) -> AnalysisResult:
        """Detect if the question has ambiguity"""
        prompt = f"""
You are an expert analyst. Analyze if the question can be answered directly from the context.

QUESTION: {question}
CONTEXT: {context}
INTENT ANALYSIS: {intent_analysis}

STEP-BY-STEP ANALYSIS:
1. First, identify the key information the question is asking for
2. Then, scan the context for that exact information
3. Check if the context provides a direct, explicit answer

DECISION RULES (Apply in order):
❌ AMBIGUOUS (has_ambiguity = true) if:
- Question uses vague references like "the project", "big project", "main project", "important project", "it", "that one" when multiple entities exist
- Question uses descriptive terms without specific names (e.g., "major project", "primary initiative", "key system")
- Context mentions multiple projects/items but question doesn't specify which one by exact name
- Context lacks the specific information needed to answer
- Context contains contradictory information
- Answer requires interpretation or assumptions about which entity is being referenced

✅ NOT AMBIGUOUS (has_ambiguity = false) if:
- Question specifically names the entity being asked about (e.g., "Odyssey project", "Project Titan")
- Only one relevant entity exists in the context that matches the question
- Context contains a direct statement that answers the question unambiguously
- The answer is explicit and requires no interpretation
- No additional context is needed

EXAMPLES:
Question: "When is the project release due?"
Context: "Project Titan is releasing tomorrow. Project Rhea will be released day after tomorrow."
→ has_ambiguity = true (Question says "the project" but context has multiple projects)

Question: "What's the status of the big project?"
Context: "Project Alpha is 90% complete and costs $2M. Project Beta is 50% complete and costs $500K."
→ has_ambiguity = true (Which project is "big"? Could refer to size, cost, or completion)

Question: "When will the main initiative be completed?"
Context: "Initiative X launches in March. Initiative Y completes in April."
→ has_ambiguity = true ("main initiative" is subjective - which one is main?)

Question: "When did the Odyssey project start?"
Context: "The Odyssey project started in 2023 and is 80% complete."
→ has_ambiguity = false (Specific project name given, context directly answers)

Question: "What is the budget for the system?"
Context: "Payment system costs $100K. Security system costs $200K. Analytics system costs $150K."
→ has_ambiguity = true (Multiple systems exist, "the system" is vague)

Question: "When will the Titan project be completed?"
Context: "The Titan project timeline shows completion by Q2 2024."
→ has_ambiguity = false (Specific project name, direct answer in context)

Now analyze the given question and context. Respond ONLY in this JSON format:

{{
    "has_ambiguity": false,
    "confidence_level": "HIGH",
    "reasoning": "Explain your decision clearly",
    "answer": "Direct answer from context or null",
    "clarifying_questions": null,
    "contradictions": null
}}
"""
        response = self._call_openai(prompt)
        return self._parse_analysis(response)

    def _parse_analysis(self, text: str) -> AnalysisResult:
        """Parse the analysis response into structured format"""
        try:
            # Clean the text and extract JSON
            cleaned_text = text.strip()
            
            # Find JSON block
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                return AnalysisResult(**data)
            else:
                # Fallback: extract from structured text
                return self._parse_structured_text(cleaned_text)
        except Exception as e:
            print(f"Parsing error: {e}")
            print(f"Raw text: {text}")
            # Conservative fallback
            return AnalysisResult(
                has_ambiguity=True,
                confidence_level="LOW",
                reasoning=f"Failed to parse response: {str(e)}",
                clarifying_questions=["Could you please rephrase your question?"]
            )

    def _parse_structured_text(self, text: str) -> AnalysisResult:
        """Fallback parsing for non-JSON responses"""
        has_ambiguity = True
        confidence_level = "LOW"
        
        # Check for ambiguity indicators first (prioritize these)
        ambiguity_indicators = [
            "multiple projects", "which project", "vague reference", "ambiguous", "unclear", 
            "multiple", "which", "clarify", "big project", "main project", "important project",
            "major project", "primary", "key system", "the system", "the project", "main initiative",
            "descriptive terms", "subjective", "could refer to"
        ]
        
        clear_answer_indicators = [
            "directly states", "explicitly answers", "clear answer", "directly answers", 
            "specifically named", "specific project name", "direct answer", "unambiguously"
        ]
        
        if any(phrase in text.lower() for phrase in ambiguity_indicators):
            has_ambiguity = True
            confidence_level = "LOW"
        # Then check for clear answer indicators
        elif any(phrase in text.lower() for phrase in clear_answer_indicators):
            has_ambiguity = False
            confidence_level = "HIGH"
        
        return AnalysisResult(
            has_ambiguity=has_ambiguity,
            confidence_level=confidence_level,
            reasoning=text,
            answer=None if has_ambiguity else "Answer found in context",
            clarifying_questions=["Please provide more specific information."] if has_ambiguity else None
        )

    def generate_clarification(self, question: str, context: str, analysis: str) -> str:
        """Generate clarifying questions when ambiguity is detected"""
        prompt = f"""
The question cannot be answered directly from the context. Provide a helpful response.

QUESTION: {question}
CONTEXT: {context}
ANALYSIS: {analysis}

Create a response that:
1. Acknowledges what information IS available
2. Explains why the question cannot be answered definitively
3. Asks specific clarifying questions
4. Is helpful and constructive

Format your response as a natural, helpful message.
"""
        return self._call_openai(prompt)

    def generate_direct_answer(self, question: str, context: str, analysis: str) -> str:
        """Generate a direct answer when no ambiguity is detected"""
        prompt = f"""
Provide a direct answer based on the context.

QUESTION: {question}
CONTEXT: {context}
ANALYSIS: {analysis}

Rules:
- Answer directly and concisely
- Only use information from the provided context
- Be specific and factual
- Reference the source information when helpful

Provide a clear, direct answer.
"""
        return self._call_openai(prompt)

    def process_question(self, question: str, context: str, is_followup: bool = False) -> str:
        """
        Main processing function that routes through the appropriate chains
        """
        if not is_followup:
            print("🔍 Analyzing intent...")
            # Step 1: Analyze intent
            intent_analysis = self.analyze_intent(question)
            print(f"Intent Analysis: {intent_analysis}\n")
            
            print("🧐 Checking for ambiguity...")
            # Step 2: Detect ambiguity
            analysis = self.detect_ambiguity(question, context, intent_analysis)
            print(f"Analysis Result: {analysis}\n")
            
            # Step 3 & 4: Route to appropriate response
            if analysis.has_ambiguity or analysis.confidence_level == "LOW":
                print("📝 Generating clarifying questions...")
                return self.generate_clarification(question, context, analysis.reasoning)
            else:
                print("✅ Providing direct answer...")
                return self.generate_direct_answer(question, context, analysis.reasoning)
        else:
            # For follow-up questions, skip ambiguity detection and go straight to answering
            print("✅ Processing follow-up answer...")
            return self.generate_direct_answer(question, context, "Follow-up question - providing direct answer")

    def interactive_session(self, initial_question: str, initial_context: str):
        """
        Handle an interactive session with potential follow-up questions
        """
        current_question = initial_question
        current_context = initial_context
        
        # Process the initial question
        response = self.process_question(current_question, current_context)
        print(f"\n🎯 Response:\n{response}")
        
        # Check if it's asking for clarification (simple heuristic)
        if any(indicator in response.lower() for indicator in [
            "which", "clarify", "specify", "could you", "please", "?", "multiple"
        ]):
            print("\n" + "="*50)
            print("💬 I need clarification. Please provide more details:")
            
            # Get follow-up response
            followup = input("Your clarification: ").strip()
            if followup and followup.lower() not in ['exit', 'quit', 'skip']:
                # Combine the follow-up with original context
                enhanced_question = f"{current_question} Specifically: {followup}"
                
                print("\n" + "="*50)
                print("🔄 Processing your clarification...")
                
                # Process with the enhanced question
                final_response = self.process_question(enhanced_question, current_context, is_followup=True)
                print(f"\n🎯 Final Response:\n{final_response}")
            else:
                print("No clarification provided. Moving to next question.")
        
        return response

# Test function specifically for the Odyssey case
def test_odyssey_case():
    qa_system = OpenAIQASystem()
    
    question = "When did the Odyssey project start?"
    context = "The Odyssey project started in 2023 and is 80% complete. The timeline shows completion by Q2 2024."
    
    print(f"Question: {question}")
    print(f"Context: {context}")
    print("="*60)
    
    response = qa_system.process_question(question, context)
    print(f"\nFinal Response: {response}")

# Enhanced test cases
def run_test_cases():
    qa_system = OpenAIQASystem()
    
    test_cases = [
        {
            "name": "Direct Answer - Odyssey Start Date",
            "question": "When did the Odyssey project start?",
            "context": "The Odyssey project started in 2023 and is 80% complete. The timeline shows completion by Q2 2024.",
            "expected": "Should provide direct answer"
        },
        {
            "name": "Ambiguous - Main Project",
            "question": "What is the status of the main project?",
            "context": "The Odyssey project started in 2023 focuses on cloud development. The Apollo project deals with front end.",
            "expected": "Should ask for clarification"
        },
        {
            "name": "Direct Answer - Completion Date",
            "question": "When will the Odyssey project be completed?",
            "context": "The Odyssey project started in 2023 and is 80% complete. The timeline shows completion by Q2 2024.",
            "expected": "Should provide direct answer"
        },
        {
            "name": "Ambiguous - Budget",
            "question": "What is the budget for Apollo?",
            "context": "The Odyssey project has a budget of $500K. The Apollo project deals with front end development.",
            "expected": "Should ask for clarification"
        },
        {
            "name": "Ambiguous - Project Release",
            "question": "When is the project release due?",
            "context": "Project titan is releasing tomorrow. Project Rhea will be released day after tomorrow.",
            "expected": "Should ask for clarification (multiple projects)"
        },
        {
            "name": "Ambiguous - Big Project",
            "question": "What's the status of the big project?",
            "context": "Project Alpha is 90% complete and costs $2M. Project Beta is 50% complete and costs $500K.",
            "expected": "Should ask for clarification (vague descriptor 'big')"
        },
        {
            "name": "Ambiguous - Main System",
            "question": "What is the budget for the main system?",
            "context": "Payment system costs $100K. Security system costs $200K. Analytics system costs $150K.",
            "expected": "Should ask for clarification (which system is 'main'?)"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*20} TEST CASE {i}: {test['name']} {'='*20}")
        print(f"Question: {test['question']}")
        print(f"Context: {test['context']}")
        print(f"Expected: {test['expected']}")
        print("-" * 80)
        
        try:
            response = qa_system.process_question(test['question'], test['context'])
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")

# Main interactive function
def main():
    print("🤖 Context-Aware Q&A System (ChatOpenAI)")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY environment variable not found.")
        print("You can either:")
        print("1. Set the OPENAI_API_KEY environment variable")
        print("2. Enter your API key when prompted")
        print("3. Pass it directly to the OpenAIQASystem constructor")
        print()
        
        manual_key = input("Enter your OpenAI API key (or press Enter to use env var): ").strip()
        if manual_key:
            api_key = manual_key
    
    print("\nChoose an option:")
    print("1. Interactive Q&A (ask your own questions)")
    print("2. Run test cases")
    print("3. Test specific Odyssey case")
    print("4. Choose different OpenAI model")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    # Model selection
    model_name = "gpt-3.5-turbo"
    if choice == "4":
        print("\nAvailable models:")
        print("1. gpt-3.5-turbo (default, cost-effective)")
        print("2. gpt-4 (more capable, higher cost)")
        print("3. gpt-4-turbo (faster GPT-4)")
        print("4. gpt-4o (latest optimized model)")
        
        model_choice = input("Choose model (1-4): ").strip()
        model_map = {
            "1": "gpt-3.5-turbo",
            "2": "gpt-4",
            "3": "gpt-4-turbo",
            "4": "gpt-4o"
        }
        model_name = model_map.get(model_choice, "gpt-3.5-turbo")
        print(f"Selected model: {model_name}")
        choice = input("\nNow choose option (1-3): ").strip()
    
    if choice == "1":
        # Interactive mode
        qa_system = OpenAIQASystem(api_key=api_key, model_name=model_name)
        
        while True:
            print("\nEnter your question and context (or 'exit' to quit):")
            
            question = input("Question: ").strip()
            if question.lower() in ['exit', 'quit']:
                break
            
            context = input("Context: ").strip()
            if not context:
                print("Please provide context for your question.")
                continue
            
            print("\n" + "="*50)
            try:
                qa_system.interactive_session(question, context)
            except Exception as e:
                print(f"Error: {e}")
            print("="*50)
    
    elif choice == "2":
        # Run test cases
        print(f"\nRunning all test cases with {model_name}...")
        run_test_cases()
    
    elif choice == "3":
        # Test specific Odyssey case
        print(f"\nTesting Odyssey case with {model_name}:")
        test_odyssey_case()
    
    else:
        print("Invalid choice. Starting interactive mode...")
        main()

if __name__ == "__main__":
    main()