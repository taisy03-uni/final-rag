import requests
import os
import time
from bs4 import BeautifulSoup

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
    
    def make_folders_court(self):
        print("Making folders for the courts...")
        for tag in self.tags_court:
            #make the folder for the court
            court = list(tag.keys())[0]
            #make folder
            folder_name = f"data/court/{court}"
            try:
                os.makedirs(folder_name)
            except FileExistsError:
                print(f"Folder {folder_name} already exists")
            # if tag value is a list, iterate through the list
            if isinstance(tag[list(tag.keys())[0]], list):
                for sub_tag in tag[list(tag.keys())[0]]:
                    subtag = list(sub_tag.keys())[0]
                    folder_name = f"data/court/{court}/{subtag}"
                    try:
                        os.makedirs(folder_name)
                    except FileExistsError:
                        print(f"Folder {folder_name} already exists")
                    for year in self.years:
                        #make folder for each year
                        folder_name = f"data/court/{court}/{subtag}/{year}"
                        try:
                            os.makedirs(folder_name)
                        except FileExistsError:
                            print(f"Folder {folder_name} already exists")
            else:
                # if tag value is not a list, make folder for each year
                for year in self.years:
                    folder_name = f"data/court/{court}/{year}"
                    try:
                        os.makedirs(folder_name)
                    except FileExistsError:
                        print(f"Folder {folder_name} already exists")
    def make_folders_tribunal(self):
        print("Making folders for tribunals...")
        for tag in self.tags_tribunals:
            court = list(tag.keys())[0]
            #make folder
            folder_name = f"data/tribunals/{court}"
            try:
                os.makedirs(folder_name)
            except FileExistsError:
                print(f"Folder {folder_name} already exists")
            
            # if tag value is a list, iterate through the list
            if isinstance(tag[list(tag.keys())[0]], list):
                for sub_tag in tag[list(tag.keys())[0]]:
                        subtag = list(sub_tag.keys())[0]
                        folder_name = f"data/tribunals/{court}/{subtag}"
                        try:
                            os.makedirs(folder_name)
                        except FileExistsError:
                            print(f"Folder {folder_name} already exists")
                        for year in self.years:
                            #make folder for each year
                            folder_name = f"data/tribunals/{court}/{subtag}/{year}"
                            try:
                                os.makedirs(folder_name)
                            except FileExistsError:
                                print(f"Folder {folder_name} already exists")
            else:
                # if tag value is not a list, make folder for each year
                for year in self.years:
                    folder_name = f"data/tribunals/{court}/{year}"
                    try:
                        os.makedirs(folder_name)
                    except FileExistsError:
                        print(f"Folder {folder_name} already exists")

    def get_all_data(self):
        print("Getting all data...")
        for tag in self.tags_court:
            #make the folder for the court
            court = list(tag.keys())[0]
            # if tag value is a list, iterate through the list
            if isinstance(tag[list(tag.keys())[0]], list):
                for sub_tag in tag[list(tag.keys())[0]]:
                    subtag = list(sub_tag.keys())[0]
                    for year in self.years:
                        self.get_court_data(court = subtag, year = year)
            else:
                for year in self.years:
                    self.get_court_data(court = court, year = year)
            print(f"Finished getting data for {court}")
        print("Starting getting tribunal data...")
        for tag in self.tags_tribunals:
            court = list(tag.keys())[0]
            # if tag value is a list, iterate through the list
            if isinstance(tag[list(tag.keys())[0]], list):
                for sub_tag in tag[list(tag.keys())[0]]:
                    subtag = list(sub_tag.keys())[0]
                    for year in self.years:
                        self.get_tribunal_data(court = subtag, year = year)
            else:
                for year in self.years:
                    self.get_tribunal_data(court = court, year = year)
            print(f"Finished getting data for {court}")
        print("Finished getting all data")

    def get_court_data(self, court, year):
        print(f"Getting data for {court} in {year}")
        page = 1
        if "%2F" in court:
            folder = f"data/court/{court.split('%2F')[0]}/{court}/{year}"
        else:
            folder = f"data/court/{court}/{year}"
        while True:
            print(f"Page {page}")
            url = f"{self.atom_url}?court={court}&from_date_2={year}&to_date_2={year}&page={page}&order=-date&per_page=50"
            print(f"URL: {url}")
            data = requests.get(url).text
            soup = BeautifulSoup(data, "xml")
            entries = soup.find_all("entry")
            print(f"Number of entries: {len(entries)}")
            if len(entries) == 0:
                print("No more entries found")
                break
            for entry in entries:
                #title = entry.find("title").text
                link = entry.find("link").get("href")
                #date_published = entry.find("published").text.split("T")[0]
                data = requests.get(f"{link}/data.xml").text
                # Save the data to a file
                file_number = link.split("/")[-1]
                with open(f"{folder}/{file_number}.xml", "w") as file:
                    try:
                        file.write(data)
                    except:
                        print(f"Error writing file {file_number}.xml")
                        breakpoint()
                print(f"Saved {file_number}.xml")

            if len(entries) < 50:
                print("Last page reached")
                break
            else:
                page += 1
    def get_tribunal_data(self, court, year):
        print(f"Getting data for {court} in {year}")
        page = 1
        if "%2F" in court:
            folder = f"data/tribunals/{court.split('%2F')[0]}/{court}/{year}"
        else:
            folder = f"data/tribunals/{court}/{year}"
        while True:
            print(f"Page {page}")
            url = f"{self.atom_url}?tribunal={court}&from_date_2={year}&to_date_2={year}&page={page}&order=-date&per_page=50"
            print(f"URL: {url}")
            data = requests.get(url).text
            soup = BeautifulSoup(data, "xml")
            entries = soup.find_all("entry")
            print(f"Number of entries: {len(entries)}")
            if len(entries) == 0:
                print("No more entries found")
                break
            for entry in entries:
                #title = entry.find("title").text
                link = entry.find("link").get("href")
                #date_published = entry.find("published").text.split("T")[0]
                data = requests.get(f"{link}/data.xml").text
                # Save the data to a file
                file_number = link.split("/")[-1]
                with open(f"{folder}/{file_number}.xml", "w") as file:
                    try:
                        file.write(data)
                    except:
                        print(f"Error writing file {file_number}.xml")
                        breakpoint()
                print(f"Saved {file_number}.xml")

            if len(entries) < 50:
                print("Last page reached")
                break
            else:
                page += 1

    def test_downloads(self):
        ## returns the number of files in the data folder
        root_dir = "data"
        results = []
        count = 0
        for dirpath, dirnames, filenames in os.walk(root_dir):
            print(f"Found directory: {dirpath}")
            print(len(filenames))
            count += len(filenames)
        print(f"Total number of files: {count}")    

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

if __name__ == "__main__":
    bot = DataDownload()
    """# Uncomment this if you are accessing this for the first time
    bot.make_folders_court()
    bot.make_folders_tribunal()
    bot.get_all_data()
    bot.test_downloads()
    #example usage of the get_file_paths method"""
    print(bot.get_file_paths(court = "ewca%2Fciv", year = "2002"))
