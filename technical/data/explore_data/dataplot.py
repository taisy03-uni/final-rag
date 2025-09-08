import matplotlib.pyplot as plt
import pandas as pd
from graphviz import Digraph


# This file is used to generate the information of the

# Yearly document data
yearly_data = {
    2000: 0, 2001: 3, 2002: 3, 2003: 2036, 2004: 2072,
    2005: 2124, 2006: 2347, 2007: 2511, 2008: 3004,
    2009: 3061, 2010: 2910, 2011: 2790, 2012: 2825,
    2013: 3025, 2014: 3110, 2015: 3257, 2016: 2862,
    2017: 2745, 2018: 2841, 2019: 2857, 2020: 3080,
    2021: 1918, 2022: 3604, 2023: 4578, 2024: 5408,
    2025: 2546
}


# Convert to DataFrame
df_yearly = pd.DataFrame(list(yearly_data.items()), columns=['Year', 'Documents'])

# Plotting
plt.figure(figsize=(12, 6))
bars = plt.bar(df_yearly['Year'], df_yearly['Documents'], color='royalblue')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 100,  
        f'{height:,}', 
        ha='center', va='bottom', fontsize=8
    )

plt.title('Number of Documents by Year')
plt.xlabel('Year')
plt.ylabel('Number of Documents')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.xticks(rotation=45)

# Show plot
plt.savefig('by_year.png', dpi=300)



#####========== TRIBUNALS TREE ==========

tribunals = Digraph('UK_Tribunals_Tree', format='png')
tribunals.attr(rankdir='TB') 
tribunals.attr(nodesep='0.2', ranksep='0.5')
tribunals.node_attr.update(
    style='filled',
    fillcolor='#354AB8',
    fontcolor='white',
    color='#354AB8',  # border color
    shape='box'
)


# Root
tribunals.node("UK Tribunals\n(5,576)")

# Top-level tribunals
tribunals.edges([
    ("UK Tribunals\n(5,576)", "Investigatory Powers (8)"),
    ("UK Tribunals\n(5,576)", "Employment Appeal (584)"),
    ("UK Tribunals\n(5,576)", "Upper Tribunal\n(2,405)"),
    ("UK Tribunals\n(5,576)", "First-tier Tribunal\n(2,575)"),
    ("UK Tribunals\n(5,576)", "Immigration Services (4)")
])

# Upper Tribunal branches
tribunals.edges([
    ("Upper Tribunal\n(2,405)", "Admin Appeals (745)"),
    ("Upper Tribunal\n(2,405)", "Immigration/Asylum (1008)"),
    ("Upper Tribunal\n(2,405)", "Lands (407)"),
    ("Upper Tribunal\n(2,405)", "Tax & Chancery (245)")
])

# First-tier Tribunal branches
tribunals.edge("First-tier Tribunal\n(2,575)", "General Regulatory (1511)")
tribunals.edge("First-tier Tribunal\n(2,575)", "Tax Chamber (1064)")

# Render Tribunals Tree
tribunals.render('uk_tribunals_tree_compact', cleanup=True)


#####========== COURTS TREE ==========

# Create a cleaner UK Courts Tree
courts = Digraph('UK_Courts_Tree', format='png')
courts.attr(rankdir='TB')  # Top to bottom for better hierarchy visualization
courts.attr(splines='ortho') 
courts.attr(nodesep='0.3', ranksep='0.5')
courts.attr(bgcolor='white')

# Define color scheme - all purple
primary_color = '#354AB8'

# Root node styling
courts.node("UK Courts\n(61,941)", 
           style='filled', 
           fillcolor=primary_color, 
           fontcolor='white',
           fontsize='16',
           shape='box',
           width='2')

# Top-level courts - Supreme level
courts.node("Supreme Court\n(912)", 
           style='filled', 
           fillcolor=primary_color, 
           fontcolor='white',
           fontsize='12',
           shape='box')

courts.node("Privy Council\n(650)", 
           style='filled', 
           fillcolor=primary_color, 
           fontcolor='white',
           fontsize='12',
           shape='box')

# Appeal level courts
courts.node("Court of Appeal\n(21,326)", 
           style='filled', 
           fillcolor=primary_color, 
           fontcolor='white',
           fontsize='12',
           shape='box')

courts.node("Civil Division (14,540)\nCriminal Division (6,786)", 
           style='filled', 
           fillcolor=primary_color, 
           fontcolor='white',
           fontsize='12',
           shape='box')


# High Court - single consolidated node

courts.node("High Court\n(36,861)", 
           style='filled', 
           fillcolor=primary_color, 
           fontcolor='white',
           fontsize='12',
           shape='box')

courts.node("Administrative (12,718)\nChancery (8,325)\nKing's Bench (5,847)\nCommercial (3,773)\nFamily (2,844)\nTech & Construction (1,845)\nPatents (679)\nSenior Costs Office (467)\nIP Enterprise (254)\nAdmiralty (105)\nMercantile (4)", 
           style='filled', 
           fillcolor=primary_color, 
           fontcolor='white',
           fontsize='12',
           shape='box',
           width='3')

# Other Courts - single consolidated node 
courts.node("Other Courts\n(2,192)", 
           style='filled', 
           fillcolor=primary_color, 
           fontcolor='white',
           fontsize='12',
           shape='box')

courts.node("Family Court (1,424)\nCourt of Protection (707)\nCounty Court (51)\nCrown Court (10)", 
           style='filled', 
           fillcolor=primary_color, 
           fontcolor='white',
           fontsize='12',
           shape='box',
           width='2.5')

# Create connections
courts.edges([
    ("UK Courts\n(61,941)", "Supreme Court\n(912)"),
    ("UK Courts\n(61,941)", "Privy Council\n(650)"),
    ("UK Courts\n(61,941)", "High Court\n(36,861)"),
    ("UK Courts\n(61,941)", "Court of Appeal\n(21,326)"),
    ("UK Courts\n(61,941)", "Other Courts\n(2,192)")
])

# Court of Appeal subdivisions
courts.edges([
    ("High Court\n(36,861)", "Administrative (12,718)\nChancery (8,325)\nKing's Bench (5,847)\nCommercial (3,773)\nFamily (2,844)\nTech & Construction (1,845)\nPatents (679)\nSenior Costs Office (467)\nIP Enterprise (254)\nAdmiralty (105)\nMercantile (4)"),
    ("Court of Appeal\n(21,326)", "Civil Division (14,540)\nCriminal Division (6,786)"),
    ("Other Courts\n(2,192)", "Family Court (1,424)\nCourt of Protection (707)\nCounty Court (51)\nCrown Court (10)"),
])

# Render the tree
courts.render('uk_courts_tree_prettier', cleanup=True)
print("Prettier UK Courts tree generated successfully!")