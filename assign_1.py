import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    """
    Load the dataset from the given file path.

    Parameters:
    file_path (str): The path to the dataset file.

    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    try:
        data = pd.read_csv("weather.csv")
        print(data.DataFrame)

        data["Average Temperature Celsius"] = (data["Avg Temp"] - 32) * 5/9
        data["Precipitation mm"] = data["Precipitation"] * 25.4

        ##Q4
        
        #if data["Station.City"] == "San Francisco":
        #    df_sf["Date"] = pd.to_datetime(data["Date"])
        #    df_sf["Temperature"] = data["Average Temperature Celsius"]
        
        df_sf = pd.DataFrame([
            (pd.to_datetime(data["Date"]), pd.to_numeric(data["Average Temperature Celsius"]))
            for _, row in data.iterrows()
            if row["Station.City"] == "San Francisco"
        ])
        df_sf.columns = ["Date", "Temperature"]

###########################

        df_sf = data[data["Station.City"] == "San Francisco"].copy()

        # Convert types
        df_sf["Date"] = pd.to_datetime(df_sf["Date.Full"])
        df_sf["Temperature"] = pd.to_numeric(df_sf["Average Temperature Celsius"])

        plt.plot(df_sf["Date"], df_sf["Temperature"], label="Average temperature in San Francisco by date in Celsius")
        plt.xlabel("Date")
        plt.ylabel("Average Temperature (°C)")
        plt.title("Average Temperature in San Francisco Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()
##########################
        wanted_cities = ["Anchorage", "Boston", "Dallas-Fort Worth"]
        df_aktxma = df[df["Station.City"].isin(wanted_cities)].copy()
        
        #df_aktxma is already just the wanted cities
        df_aktxma_temperature = df_aktxma.pivot(index=pd.to_datetime(df_aktxma['Date.Full']), columns='Station.City', values='Data.Temperature.Avg Temp')
        df_aktxma_precipitation = df_aktxma.pivot(index=pd.to_datetime(df_aktxma['Date.Full']), columns='Station.City', values='Data.Precipitation')
##########################

        df_aktxma_temperature["Date"] = pd.to_datetime(df_aktxma_temperature["Date.Full"])
        plt.plot(df_aktxma_temperature["Date"], df_aktxma_temperature["Temperature"], label="Average temperature in Anchorage in Celsius")
        plt.plot(df_aktxma_temperature["Date"], df_aktxma_temperature["Temperature"], label="Average temperature in Boston in Celsius")
        plt.plot(df_aktxma_temperature["Date"], df_aktxma_temperature["Temperature"], label="Average temperature in Dallas-Fort Worth in Celsius")
        plt.xlabel("Date")
        plt.ylabel("Average Temperature (°C)")
        plt.title("Average Temperature per City Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()
##########################
        return data.DataFrame
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

