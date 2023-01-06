import streamlit as st
st.image('PPR.jpg', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
st.subheader("PREDICTING PESTE DES PETITS RUMINANTS WITH MACHINE LEARNING")

st.write("PPR is a severe, fast-spreading disease of mainly domestic small ruminants. It is characterized by the sudden onset of depression, fever, discharges from the eyes and nose, sores in the mouth, disturbed breathing and cough, foul-smelling diarrhoea and death.")
#st.write("Clinical signs")

symptoms = ["Rectal temperature >= 40° C","Loose faeces", "Depressed", "Sleepy", "Eyes Discharge", "Nose Discharge", "Mouth Sores", "Disturbed breathing", "Cough", "Foul-smelling Diarrhoea", "Erect Hair", "Dry eyeS"]

location = st.text_input("Enter your location:")

selected_symptoms = st.multiselect("Select the symptoms:", symptoms)

if selected_symptoms:
    st.write("You have selected the following symptoms:")
    #for symptom in selected_symptoms:
        #st.write(symptom)

symptom_values = {}

for symptom in symptoms:
    if symptom in selected_symptoms:
        symptom_values[symptom] = 1
    else:
        symptom_values[symptom] = 0



st.write(symptom_values)
num_symptoms = len(symptoms)
num_selected_symptoms = len(selected_symptoms)
Comments = st.text_area("Enter your comments:")
st.write("PREDICTED RESULTS")
if num_selected_symptoms > num_symptoms / 2:
    st.write("Possible PPR case")
else:
    st.write("Not a possible PPR case")
if location:
    st.write("Your location is:", location)    

if Comments:
    st.write("Your Comments Are:", Comments)
    
results = st.button("SUBMIT ANIMAL SYMPTOMS")
st.write(results)
