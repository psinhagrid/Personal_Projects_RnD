from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any, Dict
import json
import re
from groq import Groq
from pydantic import BaseModel, Field

# Custom Groq LLM wrapper for LangChain
class GroqLLM(LLM):
    client: Any = Field(default=None)
    model_name: str = Field(default="meta-llama/llama-4-scout-17b-16e-instruct")
    temperature: float = Field(default=0.0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Groq()
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_completion_tokens=512,
            top_p=1,
            stream=False,
            stop=stop,
        )
        return completion.choices[0].message.content

# Output parser for analysis results
class AnalysisResult(BaseModel):
    has_ambiguity: bool = Field(description="Whether the question has ambiguity or context is insufficient")
    confidence_level: str = Field(description="HIGH, MEDIUM, or LOW confidence in answering")
    reasoning: str = Field(description="Explanation of the analysis")
    answer: Optional[str] = Field(description="Direct answer if no ambiguity", default=None)
    clarifying_questions: Optional[List[str]] = Field(description="List of clarifying questions", default=None)
    contradictions: Optional[List[str]] = Field(description="List of contradictions found", default=None)

class AnalysisOutputParser(BaseOutputParser[AnalysisResult]):
    def parse(self, text: str) -> AnalysisResult:
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
        # More robust fallback parsing
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

# Improved Step 2: Context Analysis and Ambiguity Detection Chain
ambiguity_detection_prompt = PromptTemplate(
    input_variables=["question", "context", "intent_analysis"],
    template="""
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
)

# Intent Analysis Chain
intent_analysis_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Analyze what specific information the user is asking for:

Question: {question}

Identify:
1. The main subject/entity being asked about
2. The specific attribute or information requested
3. Any qualifiers or conditions in the question

Provide a brief analysis in 2-3 sentences.
"""
)

# Clarification Chain
clarification_prompt = PromptTemplate(
    input_variables=["question", "context", "analysis"],
    template="""
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
)

# Direct Answer Chain
direct_answer_prompt = PromptTemplate(
    input_variables=["question", "context", "analysis"],
    template="""
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
)

class ContextAwareQASystem:
    def __init__(self):
        self.llm = GroqLLM(temperature=0.0)
        self.parser = AnalysisOutputParser()
        
        # Initialize chains
        self.intent_chain = LLMChain(
            llm=self.llm,
            prompt=intent_analysis_prompt,
            verbose=False
        )
        
        self.ambiguity_chain = LLMChain(
            llm=self.llm,
            prompt=ambiguity_detection_prompt,
            verbose=False
        )
        
        self.clarification_chain = LLMChain(
            llm=self.llm,
            prompt=clarification_prompt,
            verbose=False
        )
        
        self.answer_chain = LLMChain(
            llm=self.llm,
            prompt=direct_answer_prompt,
            verbose=False
        )
    
    def process_question(self, question: str, context: str, is_followup: bool = False) -> str:
        """
        Main processing function that routes through the appropriate chains
        """
        if not is_followup:
            print("🔍 Analyzing intent...")
            # Step 1: Analyze intent
            intent_analysis = self.intent_chain.run(question=question)
            print(f"Intent Analysis: {intent_analysis}\n")
            
            print("🧐 Checking for ambiguity...")
            # Step 2: Detect ambiguity
            ambiguity_result = self.ambiguity_chain.run(
                question=question,
                context=context,
                intent_analysis=intent_analysis
            )
            print(f"Raw ambiguity result: {ambiguity_result}\n")
            
            # Parse the ambiguity analysis
            try:
                analysis = self.parser.parse(ambiguity_result)
                print(f"Parsed Analysis: {analysis}\n")
            except Exception as e:
                print(f"Parsing error: {e}")
                return "I apologize, but I encountered an error analyzing your question. Could you please rephrase it?"
            
            # Step 3 & 4: Route to appropriate response chain
            if analysis.has_ambiguity or analysis.confidence_level == "LOW":
                print("📝 Generating clarifying questions...")
                return self.clarification_chain.run(
                    question=question,
                    context=context,
                    analysis=ambiguity_result
                )
            else:
                print("✅ Providing direct answer...")
                return self.answer_chain.run(
                    question=question,
                    context=context,
                    analysis=ambiguity_result
                )
        else:
            # For follow-up questions, skip ambiguity detection and go straight to answering
            print("✅ Processing follow-up answer...")
            return self.answer_chain.run(
                question=question,
                context=context,
                analysis="Follow-up question - providing direct answer"
            )
    
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



# Enhanced test cases
def run_test_cases():
    qa_system = ContextAwareQASystem()
    
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
    print("🤖 Context-Aware Q&A System")
    print("=" * 50)
    print("Choose an option:")
    print("1. Interactive Q&A (ask your own questions)")
    print("2. Run test cases")

    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Interactive mode
        qa_system = ContextAwareQASystem()
        
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
                # Use the new interactive session method
                qa_system.interactive_session(question, context)
            except Exception as e:
                print(f"Error: {e}")
            print("="*50)
    
    elif choice == "2":
        # Run test cases
        print("\nRunning all test cases...")
        run_test_cases()
    
    
    else:
        print("Invalid choice. Starting interactive mode...")
        main()

if __name__ == "__main__":
    main()