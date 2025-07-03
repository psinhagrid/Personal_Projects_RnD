
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
