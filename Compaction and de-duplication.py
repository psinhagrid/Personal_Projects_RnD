import os
import re
import logging
import ast
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from langchain.chat_models import ChatLiteLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

INPUT_PATH = "/Users/psinha/Documents/Utils/demo_input/sample_code.py"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeBlock:
    """Data class to hold code block information"""
    content: str
    block_type: str  # 'function', 'class', 'import', 'global'
    name: str
    start_line: int
    end_line: int
    dependencies: List[str]

class CodeParser:
    """Tool 1: Advanced code parsing and structure analysis"""
    
    def __init__(self):
        self.import_patterns = [
            r'^\s*import\s+(.+)',
            r'^\s*from\s+(.+)\s+import\s+(.+)',
        ]
    
    def parse_code_file(self, content: str) -> List[CodeBlock]:
        """Parse code into structured blocks"""
        try:
            tree = ast.parse(content)
            blocks = []
            lines = content.split('\n')
            
            # Extract imports
            imports = self._extract_imports(tree, lines)
            blocks.extend(imports)
            
            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    block = self._extract_function(node, lines)
                    if block:
                        blocks.append(block)
                elif isinstance(node, ast.ClassDef):
                    block = self._extract_class(node, lines)
                    if block:
                        blocks.append(block)
            
            # Extract global variables and other statements
            global_blocks = self._extract_global_statements(tree, lines, blocks)
            blocks.extend(global_blocks)
            
            return sorted(blocks, key=lambda x: x.start_line)
            
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            return self._fallback_parse(content)
        except Exception as e:
            logger.error(f"Error parsing code: {e}")
            return self._fallback_parse(content)
    
    def _extract_imports(self, tree: ast.AST, lines: List[str]) -> List[CodeBlock]:
        """Extract import statements"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start_line = node.lineno - 1
                end_line = node.end_lineno - 1 if node.end_lineno else start_line
                
                content = '\n'.join(lines[start_line:end_line + 1])
                
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                    name = f"import_{','.join(names)}"
                else:  # ImportFrom
                    module = node.module or ''
                    names = [alias.name for alias in node.names]
                    name = f"from_{module}_import_{','.join(names)}"
                
                imports.append(CodeBlock(
                    content=content,
                    block_type='import',
                    name=name,
                    start_line=start_line,
                    end_line=end_line,
                    dependencies=[]
                ))
        
        return imports
    
    def _extract_function(self, node: ast.FunctionDef, lines: List[str]) -> Optional[CodeBlock]:
        """Extract function definition"""
        start_line = node.lineno - 1
        end_line = node.end_lineno - 1 if node.end_lineno else start_line
        
        content = '\n'.join(lines[start_line:end_line + 1])
        
        # Extract dependencies (variables used in function)
        dependencies = self._extract_dependencies(node)
        
        return CodeBlock(
            content=content,
            block_type='function',
            name=node.name,
            start_line=start_line,
            end_line=end_line,
            dependencies=dependencies
        )
    
    def _extract_class(self, node: ast.ClassDef, lines: List[str]) -> Optional[CodeBlock]:
        """Extract class definition"""
        start_line = node.lineno - 1
        end_line = node.end_lineno - 1 if node.end_lineno else start_line
        
        content = '\n'.join(lines[start_line:end_line + 1])
        
        # Extract dependencies
        dependencies = self._extract_dependencies(node)
        
        return CodeBlock(
            content=content,
            block_type='class',
            name=node.name,
            start_line=start_line,
            end_line=end_line,
            dependencies=dependencies
        )
    
    def _extract_global_statements(self, tree: ast.AST, lines: List[str], existing_blocks: List[CodeBlock]) -> List[CodeBlock]:
        """Extract global variables and other statements"""
        global_blocks = []
        existing_lines = set()
        
        for block in existing_blocks:
            for i in range(block.start_line, block.end_line + 1):
                existing_lines.add(i)
        
        current_block = []
        start_line = None
        
        for i, line in enumerate(lines):
            if i not in existing_lines and line.strip():
                if start_line is None:
                    start_line = i
                current_block.append(line)
            else:
                if current_block:
                    content = '\n'.join(current_block)
                    if content.strip():
                        global_blocks.append(CodeBlock(
                            content=content,
                            block_type='global',
                            name=f'global_{start_line}',
                            start_line=start_line,
                            end_line=i - 1,
                            dependencies=[]
                        ))
                    current_block = []
                    start_line = None
        
        # Handle remaining block
        if current_block:
            content = '\n'.join(current_block)
            if content.strip():
                global_blocks.append(CodeBlock(
                    content=content,
                    block_type='global',
                    name=f'global_{start_line}',
                    start_line=start_line,
                    end_line=len(lines) - 1,
                    dependencies=[]
                ))
        
        return global_blocks
    
    def _extract_dependencies(self, node: ast.AST) -> List[str]:
        """Extract variable dependencies from AST node"""
        dependencies = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                dependencies.append(child.id)
        return list(set(dependencies))
    
    def _fallback_parse(self, content: str) -> List[CodeBlock]:
        """Fallback parsing using regex when AST fails"""
        blocks = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for function definition
            if re.match(r'^\s*def\s+(\w+)', line):
                block, end_i = self._extract_block_by_indentation(lines, i)
                if block:
                    match = re.match(r'^\s*def\s+(\w+)', line)
                    func_name = match.group(1) if match else f'function_{i}'
                    blocks.append(CodeBlock(
                        content=block,
                        block_type='function',
                        name=func_name,
                        start_line=i,
                        end_line=end_i,
                        dependencies=[]
                    ))
                i = end_i + 1
            
            # Check for class definition
            elif re.match(r'^\s*class\s+(\w+)', line):
                block, end_i = self._extract_block_by_indentation(lines, i)
                if block:
                    match = re.match(r'^\s*class\s+(\w+)', line)
                    class_name = match.group(1) if match else f'class_{i}'
                    blocks.append(CodeBlock(
                        content=block,
                        block_type='class',
                        name=class_name,
                        start_line=i,
                        end_line=end_i,
                        dependencies=[]
                    ))
                i = end_i + 1
            
            # Check for imports
            elif re.match(r'^\s*(import|from)\s+', line):
                blocks.append(CodeBlock(
                    content=line,
                    block_type='import',
                    name=f'import_{i}',
                    start_line=i,
                    end_line=i,
                    dependencies=[]
                ))
                i += 1
            
            else:
                i += 1
        
        return blocks
    
    def _extract_block_by_indentation(self, lines: List[str], start_idx: int) -> tuple:
        """Extract a code block based on indentation"""
        if start_idx >= len(lines):
            return None, start_idx
        
        base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        block_lines = [lines[start_idx]]
        
        i = start_idx + 1
        while i < len(lines):
            line = lines[i]
            if line.strip() == '':
                block_lines.append(line)
                i += 1
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent > base_indent:
                block_lines.append(line)
                i += 1
            else:
                break
        
        return '\n'.join(block_lines), i - 1

class CodeMinifier:
    """Tool 2: Minifies code for maximum size reduction"""
    
    def __init__(self, llm_model: str = "claude-3-sonnet-20240229", api_key: str = None):
        self.llm = ChatLiteLLM(
            model=llm_model,
            api_key=api_key
        )
        
        self.minification_prompt = PromptTemplate(
            input_variables=["code", "block_type"],
            template="""
            Please minify the following Python {block_type} to reduce its size while preserving functionality.
            
            CRITICAL REQUIREMENTS:
            1. Remove ALL comments and docstrings
            2. Use shortest possible variable names (a, b, c, etc.)
            3. Combine multiple lines where possible
            4. Remove unnecessary whitespace
            5. Use list comprehensions instead of loops where possible
            6. Use lambda functions for simple operations
            7. Remove unnecessary parentheses and brackets
            8. Use shorter syntax alternatives (e.g., x if y else z)
            9. Keep the same functionality but make it as compact as possible
            
            Original {block_type}:
            {code}
            
            Minified {block_type} (return ONLY the code, no explanations):
            """
        )
    
    def minify_code_block(self, block: CodeBlock) -> str:
        """Minify a single code block"""
        if block.block_type == 'import':
            # Don't minify imports, just clean them up
            return self._clean_imports(block.content)
        
        try:
            print(f"🔄 Minifying {block.block_type}: {block.name}")
            
            chain = LLMChain(llm=self.llm, prompt=self.minification_prompt)
            result = chain.run(code=block.content, block_type=block.block_type)
            
            # Additional post-processing
            minified = self._post_process_minification(result)
            
            print(f"✅ Minified {block.block_type}: {block.name} ({len(block.content)} -> {len(minified)} chars)")
            logger.info(f"Minified {block.block_type} {block.name}: {len(block.content)} -> {len(minified)} chars")
            
            return minified.strip()
            
        except Exception as e:
            print(f"❌ Error minifying {block.block_type} {block.name}: {e}")
            logger.error(f"Error minifying {block.block_type} {block.name}: {e}")
            # Apply basic minification as fallback
            return self._basic_minify(block.content)
    
    def _clean_imports(self, import_content: str) -> str:
        """Clean up import statements"""
        lines = import_content.split('\n')
        cleaned = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                cleaned.append(line)
        
        return '\n'.join(cleaned)
    
    def _post_process_minification(self, code: str) -> str:
        """Additional minification post-processing"""
        # Remove extra whitespace
        lines = code.split('\n')
        processed = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Remove extra spaces around operators
                line = re.sub(r'\s*([=+\-*/])\s*', r'\1', line)
                # Remove spaces after commas in function calls
                line = re.sub(r',\s+', ',', line)
                # Remove spaces in brackets
                line = re.sub(r'\[\s+', '[', line)
                line = re.sub(r'\s+\]', ']', line)
                line = re.sub(r'\(\s+', '(', line)
                line = re.sub(r'\s+\)', ')', line)
                processed.append(line)
        
        return '\n'.join(processed)
    
    def _basic_minify(self, code: str) -> str:
        """Basic minification fallback"""
        lines = code.split('\n')
        minified = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''"):
                # Remove inline comments
                if '#' in line:
                    line = line.split('#')[0].strip()
                minified.append(line)
        
        return '\n'.join(minified)

class CodeDeduplicator:
    """Tool 3: Remove duplicate code patterns and functions"""
    
    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold
    
    def deduplicate_blocks(self, blocks: List[CodeBlock]) -> List[CodeBlock]:
        """Remove duplicate code blocks"""
        print(f"🔄 Analyzing {len(blocks)} code blocks for duplicates...")
        
        # Group blocks by type
        functions = [b for b in blocks if b.block_type == 'function']
        classes = [b for b in blocks if b.block_type == 'class']
        imports = [b for b in blocks if b.block_type == 'import']
        globals = [b for b in blocks if b.block_type == 'global']
        
        # Deduplicate each type
        unique_functions = self._deduplicate_functions(functions)
        unique_classes = self._deduplicate_by_name(classes)
        unique_imports = self._merge_imports(imports)
        unique_globals = self._deduplicate_globals(globals)
        
        all_unique = unique_functions + unique_classes + unique_imports + unique_globals
        removed_count = len(blocks) - len(all_unique)
        
        print(f"✅ Removed {removed_count} duplicate/redundant blocks")
        logger.info(f"Removed {removed_count} duplicate blocks out of {len(blocks)}")
        
        return sorted(all_unique, key=lambda x: x.start_line)
    
    def _deduplicate_functions(self, functions: List[CodeBlock]) -> List[CodeBlock]:
        """Deduplicate functions based on similarity"""
        if len(functions) <= 1:
            return functions
        
        unique_functions = []
        processed = set()
        
        for i, func1 in enumerate(functions):
            if i in processed:
                continue
            
            # Find similar functions
            similar_indices = []
            for j, func2 in enumerate(functions):
                if i != j and j not in processed:
                    if self._are_functions_similar(func1, func2):
                        similar_indices.append(j)
            
            if similar_indices:
                # Keep the shortest function among similar ones
                candidates = [func1] + [functions[j] for j in similar_indices]
                shortest = min(candidates, key=lambda x: len(x.content))
                unique_functions.append(shortest)
                
                # Mark all similar functions as processed
                for j in similar_indices:
                    processed.add(j)
            else:
                unique_functions.append(func1)
            
            processed.add(i)
        
        return unique_functions
    
    def _are_functions_similar(self, func1: CodeBlock, func2: CodeBlock) -> bool:
        """Check if two functions are similar"""
        # Normalize and compare
        norm1 = self._normalize_function(func1.content)
        norm2 = self._normalize_function(func2.content)
        
        if len(norm1) == 0 or len(norm2) == 0:
            return False
        
        # Calculate similarity
        similarity = self._calculate_similarity(norm1, norm2)
        return similarity > self.similarity_threshold
    
    def _normalize_function(self, code: str) -> str:
        """Normalize function for comparison"""
        lines = code.split('\n')
        normalized = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('"""'):
                # Remove variable names and normalize structure
                line = re.sub(r'\bdef\s+\w+', 'def FUNC', line)
                line = re.sub(r'\b[a-zA-Z_]\w*\b', 'VAR', line)
                line = re.sub(r'\d+', 'NUM', line)
                line = re.sub(r'["\'].*?["\']', 'STR', line)
                normalized.append(line)
        
        return '\n'.join(normalized)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        lines1 = set(text1.split('\n'))
        lines2 = set(text2.split('\n'))
        
        intersection = len(lines1.intersection(lines2))
        union = len(lines1.union(lines2))
        
        return intersection / union if union > 0 else 0.0
    
    def _deduplicate_by_name(self, blocks: List[CodeBlock]) -> List[CodeBlock]:
        """Remove blocks with duplicate names"""
        seen_names = set()
        unique_blocks = []
        
        for block in blocks:
            if block.name not in seen_names:
                unique_blocks.append(block)
                seen_names.add(block.name)
        
        return unique_blocks
    
    def _merge_imports(self, imports: List[CodeBlock]) -> List[CodeBlock]:
        """Merge similar imports"""
        if not imports:
            return []
        
        # Group imports by module
        import_groups = {}
        for imp in imports:
            content = imp.content.strip()
            if content.startswith('import '):
                module = content.split()[1].split('.')[0]
                if module not in import_groups:
                    import_groups[module] = []
                import_groups[module].append(imp)
            elif content.startswith('from '):
                module = content.split()[1]
                if module not in import_groups:
                    import_groups[module] = []
                import_groups[module].append(imp)
        
        # Keep one import per group
        merged = []
        for group in import_groups.values():
            merged.append(group[0])  # Keep the first one
        
        return merged
    
    def _deduplicate_globals(self, globals: List[CodeBlock]) -> List[CodeBlock]:
        """Remove duplicate global statements"""
        seen_content = set()
        unique_globals = []
        
        for global_block in globals:
            normalized = global_block.content.strip()
            if normalized not in seen_content:
                unique_globals.append(global_block)
                seen_content.add(normalized)
        
        return unique_globals

class CodeOptimizationSystem:
    """Main system that orchestrates code optimization"""
    
    def __init__(self, llm_model: str = "claude-3-sonnet-20240229", api_key: str = None,
                 similarity_threshold: float = 0.9):
        self.code_parser = CodeParser()
        self.code_minifier = CodeMinifier(llm_model, api_key)
        self.code_deduplicator = CodeDeduplicator(similarity_threshold)
        
        logger.info("Code Optimization System initialized")
    
    def optimize_file(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """Optimize a single Python file"""
        try:
            print(f"\n🚀 Starting code optimization: {input_file}")
            
            # Read input file
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"📄 File loaded: {len(content)} characters")
            logger.info(f"Optimizing file: {input_file}")
            
            # Step 1: Parse code into blocks
            print(f"🔍 Parsing code structure...")
            blocks = self.code_parser.parse_code_file(content)
            print(f"📦 Parsed {len(blocks)} code blocks")
            
            # Step 2: Deduplicate blocks
            print(f"🔄 Removing duplicate code blocks...")
            unique_blocks = self.code_deduplicator.deduplicate_blocks(blocks)
            print(f"📦 Kept {len(unique_blocks)} unique blocks")
            
            # Step 3: Minify each block
            print(f"⚙️ Minifying code blocks...")
            minified_blocks = []
            
            for i, block in enumerate(unique_blocks):
                minified_content = self.code_minifier.minify_code_block(block)
                minified_blocks.append(minified_content)
                print(f"✅ Block {i+1}/{len(unique_blocks)} minified")
            
            # Step 4: Combine and final cleanup
            print(f"🔗 Combining minified code...")
            final_code = self._combine_blocks(minified_blocks)
            
            # Write to output file
            print(f"💾 Writing optimized code to: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_code)
            
            # Calculate statistics
            stats = {
                'original_length': len(content),
                'optimized_length': len(final_code),
                'original_blocks': len(blocks),
                'optimized_blocks': len(unique_blocks),
                'compression_ratio': len(final_code) / len(content),
                'blocks_removed': len(blocks) - len(unique_blocks),
                'size_reduction': len(content) - len(final_code),
                'size_reduction_percent': ((len(content) - len(final_code)) / len(content)) * 100
            }
            
            print(f"🎉 Optimization complete!")
            print(f"📊 Size: {len(content)} -> {len(final_code)} chars ({stats['size_reduction_percent']:.1f}% reduction)")
            print(f"📊 Blocks: {len(blocks)} -> {len(unique_blocks)} ({stats['blocks_removed']} removed)")
            logger.info(f"Optimization complete. Output written to: {output_file}")
            
            return stats
            
        except Exception as e:
            print(f"❌ Error optimizing file: {e}")
            logger.error(f"Error optimizing file: {e}")
            raise
    
    def _combine_blocks(self, blocks: List[str]) -> str:
        """Combine code blocks with minimal separation"""
        combined = []
        
        for i, block in enumerate(blocks):
            if block.strip():
                combined.append(block.strip())
        
        # Join with single newline to minimize space
        return '\n'.join(combined)
    
    def optimize_directory(self, input_dir: str, output_dir: str, file_pattern: str = "*.py") -> Dict[str, Any]:
        """Optimize all Python files in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files = list(input_path.glob(file_pattern))
        total_stats = {
            'files_processed': 0,
            'total_original_length': 0,
            'total_optimized_length': 0,
            'total_blocks_removed': 0,
            'total_size_reduction': 0,
            'files_stats': {}
        }
        
        print(f"🔍 Found {len(files)} Python files to optimize")
        
        for file_path in files:
            output_file = output_path / file_path.name
            try:
                stats = self.optimize_file(str(file_path), str(output_file))
                total_stats['files_processed'] += 1
                total_stats['total_original_length'] += stats['original_length']
                total_stats['total_optimized_length'] += stats['optimized_length']
                total_stats['total_blocks_removed'] += stats['blocks_removed']
                total_stats['total_size_reduction'] += stats['size_reduction']
                total_stats['files_stats'][file_path.name] = stats
                
                print(f"✅ {file_path.name} optimized successfully")
                
            except Exception as e:
                print(f"❌ Failed to optimize {file_path.name}: {e}")
                logger.error(f"Failed to optimize {file_path}: {e}")
        
        if total_stats['total_original_length'] > 0:
            total_stats['overall_compression_ratio'] = (
                total_stats['total_optimized_length'] / total_stats['total_original_length']
            )
            total_stats['overall_size_reduction_percent'] = (
                (total_stats['total_size_reduction'] / total_stats['total_original_length']) * 100
            )
        else:
            total_stats['overall_compression_ratio'] = 1.0
            total_stats['overall_size_reduction_percent'] = 0.0
        
        return total_stats

def main():
    """Demo main function"""
    # Create sample Python code with redundancy for demonstration
    sample_code = '''
import os
import sys
import re
import numpy as np
import pandas as pd

def calculate_sum(first_number, second_number):
    """Calculate the sum of two numbers"""
    # This function adds two numbers together
    result = first_number + second_number
    return result

def calculate_sum_duplicate(x, y):
    """Another function that does the same thing"""
    # Add two numbers
    total = x + y
    return total

def process_data_list(input_data):
    """Process a list of data"""
    # Check if data is valid
    if input_data is None:
        return None
    
    # Process each item in the list
    processed_list = []
    for item in input_data:
        if item > 0:
            processed_item = item * 2
            processed_list.append(processed_item)
        else:
            processed_list.append(0)
    
    return processed_list

def inefficient_calculation(number_list):
    """An inefficient calculation that can be optimized"""
    result_list = []
    for i in range(len(number_list)):
        temporary_sum = 0
        for j in range(len(number_list)):
            if i != j:
                temporary_sum = temporary_sum + number_list[j]
        result_list.append(temporary_sum)
    return result_list

class SimpleDataProcessor:
    """A simple data processor class"""
    
    def __init__(self, processor_name):
        """Initialize the processor"""
        self.name = processor_name
        self.data_list = []
    
    def add_data_item(self, data_item):
        """Add a data item to the processor"""
        self.data_list.append(data_item)
        return True
    
    def process_all_data(self):
        """Process all data items"""
        processed_data = []
        for data_item in self.data_list:
            if data_item is not None:
                if isinstance(data_item, (int, float)):
                    processed_item = data_item * 2
                    processed_data.append(processed_item)
                else:
                    processed_data.append(str(data_item))
        return processed_data

# Global configuration variables
DEFAULT_MULTIPLIER_VALUE = 2
MAXIMUM_ITERATIONS = 1000
DEBUG_MODE = True

if __name__ == "__main__":
    # Main execution block
    processor = SimpleDataProcessor("test_processor")
    processor.add_data_item(1)
    processor.add_data_item(2)
    processor.add_data_item(3)
    
    result = processor.process_all_data()
    print(f"Processed data: {result}")
'''
    
    # Create input and output directories
    os.makedirs("demo_input", exist_ok=True)
    os.makedirs("demo_output", exist_ok=True)
    
    # Write sample code to file
    with open("demo_input/sample_code.py", "w") as f:
        f.write(sample_code)
    
    # Initialize the optimization system
    claude_api_key = os.getenv("CLAUDE_API_KEY", "your_claude_api_key_here")
    
    print("🤖 Initializing Code Optimization System...")
    system = CodeOptimizationSystem(
        llm_model="claude-3-sonnet-20240229",
        api_key=claude_api_key,
        similarity_threshold=0.9
    )
    print("✅ System initialized successfully!")
    # Optimize the sample file
    try:
        stats = system.optimize_file(INPUT_PATH, "demo_output/optimized_code.py")
        
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Original size: {stats['original_length']} characters")
        print(f"Optimized size: {stats['optimized_length']} characters")
        print(f"Size ratio: {stats['compression_ratio']:.2f}")
        print(f"Original blocks: {stats['original_blocks']}")
        print(f"Optimized blocks: {stats['optimized_blocks']}")
        print(f"Blocks removed: {stats['blocks_removed']}")
        print(f"Space saved: {stats['original_length'] - stats['optimized_length']} characters")
        
        # Show optimized code
        print("\n" + "="*60)
        print("OPTIMIZED CODE")
        print("="*60)
        with open("demo_output/optimized_code.py", "r") as f:
            optimized_code = f.read()
        print(optimized_code)
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        print("\nNote: This system requires:")
        print("1. pip install langchain litellm")
        print("2. Your Claude API key set as CLAUDE_API_KEY environment variable")
        print("3. Or replace 'your_claude_api_key_here' with your actual API key")

if __name__ == "__main__":
    main()