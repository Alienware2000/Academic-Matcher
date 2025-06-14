# Step 1: Fetch the webpage HTML

import requests

url = "https://engineering.yale.edu/academic-study/departments/computer-science/research-areas"

response = requests.get(url)        # Send GET request to URL
html = response.text                # Extract the HTML text content

print(html[:1000])                  # Print the first 1000 characters just to inspect it
