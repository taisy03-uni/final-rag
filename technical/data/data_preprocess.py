import requests
import os
import time
from bs4 import BeautifulSoup
import re
import json

#This file was used to preprocess data for final evaluation

class DataDownload():
    def __init__(self):
        self.main_url = "https://caselaw.nationalarchives.gov.uk"
        self.atom_url = "https://caselaw.nationalarchives.gov.uk/atom.xml"
        self.tags_court = [
            {"uksc": "United Kingdom Supreme Court"}, 
            {"ukpc": "United Kingdom Privy Council"},
            {"ewca": [
                {"ewca%2Fciv": "Court of Appeal (Civil Division)"},
                {"ewca%2Fcrim" : "Court of Appeal (Criminal Division)"}
                ]},
            {"ewhc": [
                {"ewhc%2Fadmin": "High Court (Administrative Court)"},
                {"ewhc%2Fadmlty": "High Court (Admiralty Division)"},
                {"ewhc%2Fch": "High Court (Chancery Division)"},
                {"ewhc%2Fcomm": "High Court (Commercial Court)"},
                {"ewhc%2Ffam": "High Court (Family Division)"},
                {"ewhc%2Fipec": "High Court (Intellectual Property Enterprise Court)"},
                {"ewhc%2Fkb": "High Court (King's Bench Division)"},
                {"ewhc%2Fmercantile": "High Court (Mercantile Court)"},
                {"ewhc%2Fpat": "High Court (Patents Court)"},
                {"ewhc%2Fscco": "High Court (Senior Court Costs Office)"},
                {"ewhc%2Ftcc": "High Court (Technology and Construction Court)"}
                ]},
            {"ewcr": "Crown Court"},
            {"ewcc": "County Court"},
            {"ewfc": "Family Court"},
            {"ewcop": "Court of Protection"}]

        self.tags_tribunals = [
            {"ukiptrib": "Investigatory Powers Tribunal"},
            {"eat": "Employment Appeal Tribunal"},
            {"ukut": [
                {"ukut%2Faac": "Upper Tribunal (Administrative Appeals Chamber)"},
                {"ukut%2Fiac": "Upper Tribunal (Immigration and Asylum Chamber)"},
                {"ukut%2Flc": "Upper Tribunal (Lands Chamber)"},
                {"ukut%2Ftcc": "Upper Tribunal (Tax and Chancery Chamber)"}
                ]},
            {"ukftt": [
                {"ukftt%2Fgrc": "First-tier Tribunal (General Regulatory Chamber)"},
                {"ukftt%2Ftc": "First-tier Tribunal (Tax Chamber)"}
                ]},
            {"ukist": "Immigration Services Tribunal"}]
        self.years = [str(year) for year in range(2000, 2026)]  
        #self.make_folders_court()
        #self.make_folders_tribunal()  

    def get_file_paths(self, court="", year=""):
        """
        Returns a list of file paths based on the provided court and year parameters.
        
        Args:
            court (str): The court/tribunal identifier (e.g., "uksc", "ewca%2Fciv")
            year (str): The year as a string (e.g., "2020")
        
        Returns:
            list: A list of file paths matching the criteria
        """
        root_dir = "data"
        paths = []
        
        # Walk through all directories and files
        for dirpath, _, filenames in os.walk(root_dir):
            # Skip if no files in directory
            if not filenames:
                continue
            #skip .Ds_Store and .git directories
            if ".DS_Store" in filenames or ".git" in dirpath:
                continue
            #ignore directories that are not related to court or tribunal data
            if "court" not in dirpath and "tribunals" not in dirpath:
                continue
            # Case 1: No filters - return all files
            if not court and not year:
                #ignore any files that are not XML files
                paths.extend([os.path.join(dirpath, f) for f in filenames if f.endswith('.xml')])
                
            # Case 2: Only court filter provided
            elif court and not year:
                # Check if court is in the directory path
                if court in dirpath:
                    paths.extend([os.path.join(dirpath, f) for f in filenames if f.endswith('.xml')])
                    
            # Case 3: Only year filter provided
            elif not court and year:
                # Check if year is in the directory path
                if year in dirpath:
                    paths.extend([os.path.join(dirpath, f) for f in filenames if f.endswith('.xml')])
                    
            # Case 4: Both court and year filters provided
            else:
                # Check if both court and year are in the directory path
                if court in dirpath and year in dirpath:
                    paths.extend([os.path.join(dirpath, f) for f in filenames if f.endswith('.xml')])
        
        return paths


class FileProcessor:
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        # remove any line that begins with #judgment
        text = re.sub(r'(?m)^#judgment.*\n?', '', text)
        return text

    def get_metadata(self, soup, path):
      metadata = {"path": path}

      # FRBRWork metadata
      frbrwork = soup.find('frbrwork')
      if frbrwork:
          frbr_tags = {
              'frbruri': 'uri',
              'frbrdate': 'judgment_date',
              'frbrname': 'name',
              'frbrauthor': 'author',
              'frbrcountry': 'country',
              'frbrnumber': 'number',
              'frbrthis': 'this'
          }
          for tag_name, key in frbr_tags.items():
              tag = frbrwork.find(tag_name)
              if tag:
                  # Some have 'value', some have 'date', some have 'href'
                  value = tag.attrs.get('value') or tag.attrs.get('date') or tag.attrs.get('href')
                  if value:
                      metadata[key] = value

      # Capture proprietary UK metadata (namespaced)
      uk_tags = ['uk:court', 'uk:year', 'uk:number', 'uk:cite']
      for tag_name in uk_tags:
          tag = soup.find(tag_name)
          if tag:
              metadata[tag_name.replace('uk:', '')] = tag.get_text(strip=True)

      # Capture judges from references
      judges = [p.get('showas') for p in soup.find_all('tlcperson')]
      if judges:
          metadata['judges'] = judges

      return metadata

    def process_files(self, filepaths):
        """
        filepaths: list of paths in the bucket
        Returns a list of dicts: [{'metadata': ..., 'text': ...}, ...]
        """
        results = []
        i = 0
        for path in filepaths:
            i+=1
            print(f"{i}/{len(filepaths)}")
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, "lxml")  # parse XML
                    metadata = self.get_metadata(soup, path)
                    # Get main text (adjust selector depending on your XML structure)
                    text_tag = soup.find('Text') or soup  # fallback to whole document
                    text = text_tag.get_text() if text_tag else ''
                    clean = self.clean_text(text)

                    results.append({
                        'metadata': metadata,
                        'text': clean
                    })
            except Exception as e:
                print(f"Error reading file {path}: {e}")
                return None

        return results
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        #remove any line that begins with #judgment
        text = re.sub(r'(?m)^#judgment.*\n?', '', text)
        return text


if __name__ == "__main__":
    #bot = DataDownload()
    #all_files = bot.get_file_paths()
    #print("got the paths")
    #processor = FileProcessor()
    #results = processor.process_files(all_files)
    #with open("data/preprocessed/output.json", "w", encoding="utf-8") as f:
        #json.dump(results , f, ensure_ascii=False, indent=4)
    #print("Saved to output.json")

    import ijson
    import json

    input_path = "data/preprocessed/output.json"
    output_path = "data/preprocessed/ready.json"

    with open(input_path, "r", encoding="utf-8") as infile, \
        open(output_path, "w", encoding="utf-8") as outfile:
        
        # Start JSON array
        outfile.write("[\n")
        first = True
        
        # Stream parse each object from the big array
        for i, item in enumerate(ijson.items(infile, "item")):
            metadata = item.get("metadata", {})
            text = item.get("text", "")
            
            record = {
                "_id": metadata.get("uri", ""),
                "text": text,
                "cite": metadata.get("cite", ""),
                "file_path": metadata.get("path", ""),
                "judgment_date": metadata.get("judgment_date", ""),
                "name": metadata.get("name", ""),
                "author": metadata.get("author", ""),
                "judges": metadata.get("judges", ""),
                "uri": metadata.get("uri", "")
            }
            
            # Write with commas between JSON objects
            if not first:
                outfile.write(",\n")
            json.dump(record, outfile, ensure_ascii=False)
            first = False
            
            if i % 1000 == 0:  # progress log
                print(f"Processed {i} records")
        
        # End JSON array
        outfile.write("\n]")
    
