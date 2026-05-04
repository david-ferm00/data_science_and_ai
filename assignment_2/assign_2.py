import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile as zf
import os
from bs4 import BeautifulSoup

def extract_files(zipfilename):
    """Extract a zip file and return the contents as a dictionary.

    The dictionary returned will have the year as the key (from 2013 
    to 2024) and values are HTML files stored as strings.

    Parameters:
    zipfilename -- name of the zip file to extract
    """
    with zf.ZipFile(zipfilename, 'r') as z_object:
        z_object.extractall()
        data = {}
        for fileName in z_object.namelist():
            if fileName.endswith('.html'):
                with open(fileName, 'r', encoding='utf-8') as f:
                    #get the year
                    fileName = fileName.replace(".html", "")
                    year = int(fileName.split('_')[-1])

                    #put into dict
                    data[year] = f.readlines()
                    data[year] = "\n".join(data[year])
    z_object.close()
    return data

data = extract_files('gothenburg_sold_apartments.zip')

######

def extract_announcements(page):
    """
    Extract all announcements from a single page (the entire page is supplied as a string)
    Returns the announcements as a Pandas dataframe.
    The dataframe shall have one row per announcement in the HTML page and the following columns:
    - address
    - date
    - district
    - municipality
    - price
    - area
    - rooms
    - floor

    Parameters:
    page -- content of the HTML page as a string
    """
    result = pd.DataFrame(columns=['address', 'date', 'district', 'municipality', 'price', 'area', 'rooms', 'floor'])
    soup = BeautifulSoup(page,features='html.parser') # this will suppress a warning about not specifying the default parser
    #reference = pd.read_csv("reference_2014.csv", encoding="utf-8")
    for property in soup.find_all("div", "property-card") :
        
        address= property.find("h3", "property-title").text.replace("[", "").replace("]", "")
        date = None
        district = None
        municipality = None
        price = None
        area = None
        rooms = None
        floor = None

        for detail in property.find_all("p"):
            if detail.text.startswith("datum:"):
                date = pd.to_datetime(detail.text.replace("datum:", "").replace("januari", "January").replace("februari", "February").replace("mars", "March").replace("maj", "May").replace("juni", "June").replace("juli", "July").replace("augusti", "August").replace("oktober", "October"))
                date = date.strftime("%Y-%m-%d")
                
            if detail.text.startswith("pris:"):
                price = int(detail.text.replace("pris:", "").replace("kr", "").replace(" ", "").strip())

            if detail.text.startswith("storlek:"):
                try:
                    area = float(detail.text.replace("½", ".5").replace("storlek:", "").replace("m²", "").replace(",", ".").strip())
                except ValueError:
                    pass

            if detail.text.startswith("våning:"):
                if "BV" in detail.text:
                    floor = 0
                else:
                    try:
                        floor = float(detail.text.replace("våning: ", "").replace("vån ", "").replace("½", ".5").replace(",", "."))
                    except ValueError:
                        pass

            if detail.text.startswith("rum:"):
                try:
                    rooms = float(detail.text.replace("rum:", "").replace("rum", "").replace("½", ".5").replace(",", ".").strip())
                except ValueError:
                    pass

            if detail.text.startswith("område:"):
                district_and_municipality = detail.text.replace("område: ", "").split(" · ")
                if len(district_and_municipality) > 2:
                    district = district_and_municipality[1]
                municipality = district_and_municipality[-1]

        row = pd.Series([
            address,
            date,
            district,
            municipality,
            price,
            area,
            rooms,
            floor,
        ])
        row.index = result.columns
        result.loc[len(result)] = row

        #temp_reference = reference.iloc[len(result)-1, 1:9]
        #print(row)
        #print(reference)
        #if not row.equals(temp_reference):
        #    print("Mismatch in row " + str(len(result)-1))
    return result

##############

data = extract_files('gothenburg_sold_apartments.zip')
data = extract_announcements(data[2023])
df_2023_gothenburg = data[data["municipality"] == "Göteborg"].copy()

##############

def five_point_summary(s):
    """
    Computes the five-point summary of the series s
    Returns a new series that has the following values (as index):
    - min
    - Q1
    - median
    - Q3
    - max

    Parameters:
    s -- The series to compute the summary for
    """
    holder = s.tolist()
    holder.sort()

    summary = pd.Series(index=['min', 'Q1', 'median', 'Q3', 'max'])
    summary = {
        'min': holder[0],
        'Q1': holder[len(holder) // 4],
        'median': holder[len(holder) // 2],
        'Q3': holder[3 * len(holder) // 4],
        'max': holder[-1]
    }

    return summary

price_2023_gothenburg_five_point_summary = five_point_summary(df_2023_gothenburg["price"])

#######
_, ax = plt.subplots() # axis for plotting histogram

counts, bins, _ = ax.hist(df_2023_gothenburg["price"], bins = int(np.ceil(1 + np.log2(len(df_2023_gothenburg["price"])))), histtype = 'bar')

loc = np.arange(0, 20000000 + 2500000/2, 2500000)
labels = ["0.0", "2.5", "5.0", "7.5", "10.0", "12.5", "15.0", "17.5", "20.0"]
plt.xticks(loc, labels)
plt.xlabel("Price (MSEK)")
plt.ylabel("Count")
plt.title("Histogram over Prices in Gothenburg 2023")
plt.grid(True)
plt.show()

#######

_, ax = plt.subplots(figsize=(10, 6))#adjust figsize as needed

unique_room_numbers = df_2023_gothenburg["rooms"].unique()
unique_room_numbers = [x if x != None else -1 for x in unique_room_numbers]
unique_room_numbers.sort()
unique_room_numbers = [x if x != -1 else None for x in unique_room_numbers]

color_assignment = []
for apartment in df_2023_gothenburg["rooms"]:
    color_assignment.append(unique_room_numbers.index(apartment))

unique_room_numbers = [x if x != None else "None" for x in unique_room_numbers]

scatter = ax.scatter(df_2023_gothenburg["area"], df_2023_gothenburg["price"], c=color_assignment, cmap='tab20c')

plt.xlabel("Area in M²")
plt.ylabel("Price in MSEK")
plt.title("Scatterplot of Area vs Price in Gothenburg 2023")

handles, _ = scatter.legend_elements()
plt.legend(handles, unique_room_numbers, title="Number of Rooms", loc="best", ) # adjust legend position as needed

plt.grid(True)
plt.show()

#########

df_2023_gothenburg_w_psqm = df_2023_gothenburg.copy()

series = []
for index in df_2023_gothenburg_w_psqm.index:
    area = df_2023_gothenburg_w_psqm["area"][index]
    price = df_2023_gothenburg_w_psqm["price"][index]
    if area != 0 and area != None and price != None:
        series.append(price / area)
    else:
        series.append(None)
df_2023_gothenburg_w_psqm["price_per_sqm"] = series

########

unique_room_numbers = df_2023_gothenburg_w_psqm["rooms"].unique()
unique_room_numbers = [x if x != None else -1 for x in unique_room_numbers]
unique_room_numbers.sort()
unique_room_numbers = [x if x != -1 else None for x in unique_room_numbers]

axis_labels = [i for i in range(1, len(unique_room_numbers))]

_, ax = plt.subplots() # axis for plotting
barchart = ax.bar(axis_labels, df_2023_gothenburg_w_psqm.groupby(["rooms"])["price_per_sqm"].mean(), color='blue', width=0.4)
plt.xticks(axis_labels, unique_room_numbers[1:])
plt.xlabel("Number of rooms")
plt.ylabel("Average price per square meter (SEK/m²)")
plt.title("Average Price per Square Meter by Number of Rooms in Gothenburg 2023")

plt.grid(True)
plt.show()

###########

series_price_per_sqm_5_cheapest = df_2023_gothenburg_w_psqm.groupby("district")["price_per_sqm"].mean().sort_values(ascending=True).head(5)
series_price_per_sqm_5_most_expensive = df_2023_gothenburg_w_psqm.groupby("district")["price_per_sqm"].mean().sort_values(ascending=False).head(5).sort_values(ascending=True)

data = pd.concat([series_price_per_sqm_5_cheapest, series_price_per_sqm_5_most_expensive])

axis_labels = [i for i in range(1, len(data)+1)]

_, ax = plt.subplots(figsize=(7, 7)) # axis for plotting
barchart = ax.bar(axis_labels, data, color='blue', width=0.4)

ax.set_xticks(axis_labels)
ax.set_xticklabels(data.index, rotation=90)

plt.xlabel("District")
plt.ylabel("Average price per square meter (SEK/m²)")
plt.title("Average Price per Square Meter by District in Gothenburg 2023")

plt.grid(True)
plt.show()

