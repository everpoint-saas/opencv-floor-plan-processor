"""
Example usage of the FloorPlanProcessor
"""

import os
import sys
from floor_plan_processor import FloorPlanProcessor


def main():
    # Initialize processor with workflow configuration
    processor = FloorPlanProcessor('workflow.json')
    
    # Example 1: Process a PDF file
    pdf_path = 'example.pdf'  # Replace with your PDF path
    
    if os.path.exists(pdf_path):
        print(f"Processing PDF: {pdf_path}")
        
        try:
            # Process the first page
            results = processor.process_pdf(pdf_path, page_number=0)
            
            # Save results
            output_dir = 'output'
            processor.save_results(results, output_dir)
            
            print(f"Processing complete! Results saved to {output_dir}")
            print(f"- Building outline found: {results['building_outline'] is not None}")
            print(f"- Number of rooms extracted: {len(results['rooms'])}")
            print(f"- Number of optimized rooms: {len(results['optimized_rooms'])}")
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
    else:
        print(f"PDF file not found: {pdf_path}")
        print("Please provide a valid PDF path")
    
    # Example 2: Process an image directly
    # import cv2
    # image = cv2.imread('floor_plan.png')
    # results = processor.process_image(image)
    # processor.save_results(results, 'output_image')


if __name__ == "__main__":
    main()