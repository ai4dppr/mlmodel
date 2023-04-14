import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import geocoder
import folium
from streamlit_folium import st_folium, folium_static
from openpyxl import load_workbook
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import pydeck as pdk
from openpyxl import Workbook
from geopy.geocoders import Nominatim
import datetime
import pandas as pd
import streamlit as st
import pydeck as pdk
from geopy.geocoders import Nominatim
import datetime
import pandas as pd
import streamlit as st
import pydeck as pdk
from geopy.geocoders import Nominatim
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# App title and description
st.image("PPR.jpg")
st.subheader("PPR, CCPP and Blue Tongue Disease Surveillance App")
st.write("This app helps to report and monitor the spread of PPR, CCPP and BlueTongue disease in sheep and goats.")

st.sidebar.title('Contact Information')
name = st.sidebar.text_input('Name')
email = st.sidebar.text_input('Email')
phone_number = st.sidebar.text_input('Phone Number')



# Define the input feature
st.sidebar.title('Clinical Symptoms')
def user_input_features():
    df = pd.read_csv('Updated2.csv')
    #st.sidebar.title('Clinical Symptoms')
    
    
    salivation = st.sidebar.selectbox('Salivation', ['no', 'yes'])
    difficult_in_walking = st.sidebar.selectbox('Difficult in Walking', ['no', 'yes'])
    Morbidity = st.sidebar.selectbox('Morbidity', ['no', 'yes'])
    Depression = st.sidebar.selectbox('Depression', ['no', 'yes'])
    Anorexia = st.sidebar.selectbox('Anorexia', ['no', 'yes'])
    Rough_hair = st.sidebar.selectbox('Rough Hair', ['no', 'yes'])
    Coughing = st.sidebar.selectbox('Coughing', ['no', 'yes'])
    Congested_mucus_membrane = st.sidebar.selectbox('Congested Mucus Membrane', ['no', 'yes'])
    Mortality = st.sidebar.selectbox('Mortality', ['no', 'yes'])
    
    
    temp = st.sidebar.slider('Temperature', 35.0, 41.0, 38.0, 0.1)
    nasal_discharge = st.sidebar.selectbox('Nasal Discharge', ['no', 'yes'])
    diarrhea = st.sidebar.selectbox('Diarrhea', ['no', 'yes'])
    difficult_breathing = st.sidebar.selectbox('Difficult Breathing', ['no', 'yes'])
    age = st.sidebar.selectbox('Age (above 6 months)', ['no', 'yes'])
    eye_discharge = st.sidebar.selectbox('Eye Discharge', ['no', 'yes'])
    oral_nasal_lesion = st.sidebar.selectbox('Oral/Nasal Lesion', ['no', 'yes'])
    animal = st.sidebar.selectbox('Animal (goat/sheep)', ['sheep', 'goat'])
    sex = st.sidebar.selectbox('Sex', ['female', 'male'])
  
  
    external_features = {
        'salivation': 1 if salivation == "yes" else 0,
        'difficult_in_walking': 1 if difficult_in_walking == "yes" else 0,
        'Morbidity': 1 if Morbidity == "yes" else 0,
        'Depression': 1 if Depression == "yes" else 0,
        'Anorexia': 1 if Anorexia == "yes" else 0,
        'Rough_hair': 1 if Rough_hair == "yes" else 0,
        'Coughing': 1 if Coughing == "yes" else 0,
        'Congested_mucus_membrane': 1 if Congested_mucus_membrane == "yes" else 0,
        'Mortality': 1 if Mortality == "yes" else 0,
    }

  
  
  
    data = {
        'temp': 1 if temp >= 38 else 0,
        'nasal_discharge': 1 if nasal_discharge == 'yes' else 0,
        'diarrhea': 1 if diarrhea == 'yes' else 0,
        'difficult_breathing': 1 if difficult_breathing == 'yes' else 0,
        'age': 1 if age == 'yes' else 0,
        'eye_discharge': 1 if eye_discharge == 'yes' else 0,
        'oral_nasal_lesion': 1 if oral_nasal_lesion == 'yes' else 0,
        'animal': 1 if animal == 'goat' else 0,
        'sex': 1 if sex == 'male' else 0,
        }  
    features = pd.DataFrame(data, index=[0])        
    return features, external_features  
 
df, external_features = user_input_features()





#st.write(df)

import pickle
# Load the saved model
with open('random_forest_model.pkl', 'rb') as f:
    loaded_rf = pickle.load(f)

# Use the loaded model to make predictions
prediction = loaded_rf.predict(df)
prediction_proba = loaded_rf.predict_proba(df)

#predict the test data
#pred = rf.predict(df)

#prediction = rf.predict(df)
#prediction_proba = rf.predict_proba(df)

for i, prob in enumerate(prediction_proba):
    print(f"disease probability {i+1}: {prob[0]*100:.2f}% ")


# Adding dummy latitude and longitude columns for demonstration
#data['latitude'] = np.random.uniform(-90, 90, size=len(data))
#data['longitude'] = np.random.uniform(-180, 180, size=len(data))


st.subheader('Prediction Probability')

result_text = ""

if st.button("Predict"):
    if prediction == 1 and df.iloc[0]['animal'] == 0 and external_features['salivation'] == 1 and external_features['difficult_in_walking'] == 1 and external_features['Morbidity'] == 1 and external_features['Depression'] == 1:
        result_text = f"There is {prob[1]*100:.2f}% chance that The selected animal is suffering from Bluetongue"
    elif prediction == 1 and df.iloc[0]['animal'] == 1 and external_features['Morbidity'] == 1 and external_features['Mortality'] == 1 and external_features['Depression'] == 1 and external_features['Anorexia'] == 1 and external_features['Rough_hair'] == 0 and external_features['Coughing'] == 1:
        result_text = f"There is {prob[1]*100:.2f}% chance that The selected animal is suffering from CCPP"
    elif prediction == 1  and external_features['salivation'] == 0 and external_features['difficult_in_walking'] == 0:
        result_text = f"There is {prob[1]*100:.2f}% chance that The selected animal is suffering from PPR"
    else:
        result_text = f"There is {prob[0]*100:.2f}% chance that The selected animal is not suffering from any of the diseases"

    st.write(result_text)

# Reporting form
st.subheader("Report a Suspected PPR Case")




def get_latitude_longitude(address):
    geolocator = Nominatim(user_agent="your_app_name")
    try:
        location = geolocator.geocode(address, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except GeocoderTimedOut:
        return None, None




with st.form("user_input_data.xlsx"):
    address = st.text_input("Enter your location (address or place name)")
    num_affected_animals = st.number_input("Number of affected animals", value=0)
    clinical_signs = st.text_area("Describe the observed clinical signs")
    save_submit_button = st.form_submit_button("Save and Submit")
    prediction = result_text 
    df = pd.DataFrame({ "Address": [address], "Number of affected animals": [num_affected_animals], "clinical_signs": [clinical_signs], "prediction": [prediction], "Timestamp": [datetime.datetime.now()]})
    #include clinical signs which were used to predict the result
    
       
    

    latitude, longitude = get_latitude_longitude(address)

    input_data = pd.DataFrame({
        "Address": [address],
        "Number of affected animals": [num_affected_animals],
        "clinical_signs": [clinical_signs],
        "prediction": [prediction],
        "Timestamp": [datetime.datetime.now()],
        "latitude": [latitude],
        "longitude": [longitude],
    })

    try:
        existing_data = pd.read_excel("user_input_data.xlsx")
    except FileNotFoundError:
        existing_data = pd.DataFrame(columns=["Address", "Number of affected animals", "Timestamp", "latitude", "longitude", "prediction", "clinical_signs"])

    updated_data = existing_data.append(input_data, ignore_index=True)
    updated_data.to_excel("user_input_data.xlsx", index=False)

 
from PIL import Image
image_upload = st.file_uploader("Upload images or videos", type=["png", "jpg", "jpeg", "mp4"])
if image_upload is not None:
    # Create folder if it does not exist
    folder_path = 'Images'
    os.makedirs(folder_path, exist_ok=True)

    # Save the uploaded file to the specified folder
    with open(os.path.join(folder_path, image_upload.name), 'wb') as f:
        f.write(image_upload.getbuffer())

    # Display a success message
    st.success(f"File {image_upload.name} saved to {folder_path}")


from geopy.geocoders import Nominatim

# Convert address to latitude and longitude
geolocator = Nominatim(user_agent="PPR_app")
location = geolocator.geocode(address)

if location:
    center_data = pd.DataFrame({
        'latitude': [location.latitude],
        'longitude': [location.longitude],
    })
else:
    st.write("Location not found. Using default coordinates.")
    center_data = pd.DataFrame({
        'latitude': [-6.3728253],  # Default latitude value
        'longitude': [34.8924826],  # Default longitude value
    })

# Data visualization
st.subheader("Veterinary Doctor Location")

# Create a circle with a 100 km radius
circle_radius = 100 * 500  # Convert to meters

layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data=center_data,
        get_position=["longitude", "latitude"],
        get_radius=5000,
        get_fill_color=[255, 0, 0],
        pickable=True,
        auto_highlight=True,
    ),
    pdk.Layer(
        "CircleLayer",
        data=center_data,
        get_position=["longitude", "latitude"],
        get_radius=circle_radius,
        get_stroke_color=[255, 0, 0],
        get_fill_color=[255, 0, 0, 0],  # Set alpha value to 0 for no fill color
        pickable=True,
        auto_highlight=True,
    ),
]

initial_view_state = pdk.ViewState(
    latitude=center_data.latitude[0],
    longitude=center_data.longitude[0],
    zoom=6,  # Increase this value for a more zoomed-in view
)

st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v11", initial_view_state=initial_view_state, layers=layers))


st.subheader("PPR Outbreak Map")
st.write("""The Outbreak Map displays the locations of 
         confirmed  outbreaks, helping researchers, veterinarians, and policymakers to better 
         understand the spatial 
         distribution of the disease and identify areas with a high concentration of cases.""")
import pandas as pd
import pydeck as pdk
from geopy.geocoders import Nominatim

# Load saved locations from .xls file
locations_data = pd.read_excel("user_input_data.xlsx")

# Drop rows with missing values in latitude and longitude columns
locations_data.dropna(subset=['latitude', 'longitude'], inplace=True)

# Round latitude and longitude values to 6 decimal places
locations_data['latitude'] = locations_data['latitude']
locations_data['longitude'] = locations_data['longitude']

# Select only the latitude and longitude columns
locations_data = locations_data[['latitude', 'longitude']]



# Convert address to latitude and longitude
geolocator = Nominatim(user_agent="PPR_app")
location = geolocator.geocode(address)

if location:
    center_data = pd.DataFrame({
        'latitude': [location.latitude],
        'longitude': [location.longitude],
    })
else:
    st.write("Location not found. Using default coordinates.")
    center_data = pd.DataFrame({
        'latitude': [-6.3728253],  # Default latitude value
        'longitude': [34.8924826],  # Default longitude value
    })

# Create a circle with a 100 km radius
circle_radius = 100 * 500  # Convert to meters

layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data=center_data,
        get_position=["longitude", "latitude"],
        get_radius=6000,
        get_fill_color=[255, 0, 0],
        pickable=True,
        auto_highlight=True,
    ),
    pdk.Layer(
        "ScatterplotLayer",
        data=locations_data,
        get_position=["longitude", "latitude"],
        get_radius=5500,
        get_fill_color=[255, 0, 0],
        pickable=True,
        auto_highlight=True,
    ),
    pdk.Layer(
        "CircleLayer",
        data=center_data,
        get_position=["longitude", "latitude"],
        get_radius=circle_radius,
        get_stroke_color=[255, 0, 0],
        get_fill_color=[255, 0, 0, 0],  # Set alpha value to 0 for no fill color
        pickable=True,
        auto_highlight=True,
    ),
]

initial_view_state = pdk.ViewState(
    latitude=center_data.latitude[0],
    longitude=center_data.longitude[0],
    zoom=5,
)

st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v11", initial_view_state=initial_view_state, layers=layers))





