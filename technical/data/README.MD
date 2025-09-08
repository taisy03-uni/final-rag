# Data Folder Overview

This folder contains all scripts and resources related to downloading, preprocessing, and exploring the dataset obtained from the National Archives. Additioanlly this is where the Tast2025 COILLEE Dataset is held and the chunked data

## Folder Structure

data/
├── data_preprocess.py     #preprocess the raw data for final evaluation
├── data_retrieval.py      #Sownloading relevant data from the National Archives.  
├── explorer.py            #Data Explorer for National Archives
├── explore_data/          #Folder for Data Information and Plot 



Go to https://caselaw.nationalarchives.gov.uk/re-use-find-case-law-records#:~:text=The%20Open%20Justice%20licence%20allows,that%20apply%20under%20this%20licence. and apply for the license. 

Once you obtain the license you are able to use the data in this project. 

```python
if __name__ == "__main__":
    bot = DataDownload()
    
    # Uncomment the following lines if accessing for the first time
    # bot.make_folders_court()
    # bot.make_folders_tribunal()
    # bot.get_all_data()
    # bot.test_downloads()
    
    # Example usage of the get_file_paths method
    print(bot.get_file_paths(court="ewca%2Fciv", year="2002"))