from data.data_retrieval import DataDownload
from bs4 import BeautifulSoup
import re
import random
import json
import google.generativeai as genai
from pathlib import Path

# Configure Gemini - make sure you have your API key set up
genai.configure(api_key='')
model = genai.GenerativeModel("gemini-2.5-flash")

def get_text(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'xml')  # Use 'xml' parser for XML files
            # Extract metadata
            # Get the main text content (excluding metadata)
            text = soup.get_text()
            text = re.sub(r'\n{2,}', '\n', text)
            text = re.sub(r' +', ' ', text)
            text = text.strip()
            #remove any line that begins with #judgment
            text = re.sub(r'(?m)^#judgment.*\n?', '', text)
            return text
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

def generate_summary_and_variant(original_text, path):
    try:
        # Prompt to get summary of original case
        summary_prompt = f"""
        Case text:
        {original_text} 

        Please provide a concise summary of the following legal case. 
        Focus on the key facts, legal issues, and outcome.
        I want you to otput you response in a json format as follows:
        casesummary: "The cases summary"
        outcome: "outcome of the case"
        quesstion: "What was the issue"
        citations: ["The citations mentioned in the case", why they were mentioned]
        
        """
        
        original_summary = model.generate_content(summary_prompt).text
        
        # Prompt to create a similar but modified case
        variant_prompt = f"""
        Based on the following case summary, create a new fictional case that is similar 
        but with modified details. Change at least 3 key facts (like names, dates, 
        locations, specific amounts, etc.) while keeping the core legal issues similar.
        
        Original summary:
        {original_summary}
        
        Please provide:
        1. A summary of the new fictional case
        2. A list of the specific changes you made from the original
        """
        
        variant_response = model.generate_content(variant_prompt).text
        
        # Try to split the response into summary and changes
        if "Summary of the new fictional case:" in variant_response:
            parts = variant_response.split("Summary of the new fictional case:")
            new_summary = parts[1].split("List of specific changes:")[0].strip()
            changes = parts[1].split("List of specific changes:")[1].strip() if len(parts) > 1 else "Not specified"
        else:
            new_summary = variant_response
            changes = "Changes not explicitly listed"
        
        return {
            "original_summary": original_summary,
            "new_summary": new_summary,
            "changes_made": changes,
            "original_file_path": str(path)
        }
        
    except Exception as e:
        print(f"Error generating summary/variant for {path}: {e}")
        return None

def main():
    data = DataDownload()
    paths = data.get_file_paths()
    
    # Select 20 random cases (or fewer if there aren't 20)
    sample_size = min(20, len(paths))
    selected_paths = random.sample(paths, sample_size)
    
    evaluation_data = []
    print("selected paths: ",  selected_paths)
    for path in selected_paths:
        print(f"Processing {path}...")
        text = get_text(path)
        print(text)
        breakpoint()
"""        
        if text:
            result = generate_summary_and_variant(text, path)
            if result:
                evaluation_data.append(result)
    
    # Save results
    output_file = "evaluation_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, indent=2)
    
    print(f"Successfully created evaluation dataset with {len(evaluation_data)} cases. Saved to {output_file}")"""

if __name__ == "__main__":
    main()