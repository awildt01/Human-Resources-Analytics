#Bibliotheken
import pandas as pd
import streamlit as st
from minio import Minio
import joblib
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model

#Data Lake
client = Minio(
        "localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False
    )

#Klassifikation,dataset und cluster.
client.fget_object("curated","model.pkl","model.pkl")
client.fget_object("curated","dataset.csv","dataset.csv")
client.fget_object("curated","cluster.joblib","cluster.joblib")

var_model = "model"
var_model_cluster = "cluster.joblib"
var_dataset = "dataset.csv"

#laden von Modellen .
model = load_model(var_model)
model_cluster = joblib.load(var_model_cluster)

#carregando o conjunto de dados.
dataset = pd.read_csv(var_dataset)

print (dataset.head())

# title
st.title("Human Resource Analytics")

# Untertitel
st.markdown("Diese Daten-App zeigt die Maschinelle Learning Lösung des Projekt Human Resource Analytics.")

# anzeigen die Datengruppe
st.dataframe(dataset.head())

# Arbeitsgruppe.
kmeans_colors = ['green' if c == 0 else 'red' if c == 1 else 'blue' for c in model_cluster.labels_]

st.sidebar.subheader("Mitarbeiterattribute definieren")

# Atributten die ausgesucht werden
satisfaction = st.sidebar.number_input("satisfaction", value=dataset["satisfaction"].mean())
evaluation = st.sidebar.number_input("evaluation", value=dataset["evaluation"].mean())
averageMonthlyHours = st.sidebar.number_input("averageMonthlyHours", value=dataset["averageMonthlyHours"].mean())
yearsAtCompany = st.sidebar.number_input("yearsAtCompany", value=dataset["yearsAtCompany"].mean())

# Taste
btn_predict = st.sidebar.button("klassifikation")

# Taste gedrückt
if btn_predict:
    data_teste = pd.DataFrame()
    data_teste["satisfaction"] = [satisfaction]
    data_teste["evaluation"] =	[evaluation]    
    data_teste["averageMonthlyHours"] = [averageMonthlyHours]
    data_teste["yearsAtCompany"] = [yearsAtCompany]
    
    # anzeigen von den Daten   
    print(data_teste)

    #vorhersage
    result = predict_model(model, data=data_teste)
    
    st.write(result)


    fig = plt.figure(figsize=(10, 6))
    plt.scatter( x="satisfaction"
                ,y="evaluation"
                ,data=dataset[dataset.turnover==1],
                alpha=0.25,color = kmeans_colors)

    plt.xlabel("Satisfaction")
    plt.ylabel("Evaluation")

    plt.scatter( x=model_cluster.cluster_centers_[:,0]
                ,y=model_cluster.cluster_centers_[:,1]
                ,color="black"
                ,marker="X",s=100)
    
    plt.scatter( x=[satisfaction]
                ,y=[evaluation]
                ,color="yellow"
                ,marker="X",s=300)

    plt.title("Mitarbeitergruppen - Zufriedenheit vs Bewertung.")
    plt.show()
    st.pyplot(fig) 