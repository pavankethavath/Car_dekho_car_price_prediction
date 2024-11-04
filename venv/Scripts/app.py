# importing required Libraries 
import streamlit as st
import pickle
import pandas as pd


#Cities with logos/icons 
cities = {
    "Delhi": "C:/Users/pavan/Desktop/Capestone_projects/Car_dekho_car_price_prediction/City logos/Delhi.png",
    "Bangalore": "C:/Users/pavan/Desktop/Capestone_projects/Car_dekho_car_price_prediction/City logos/Bangalore.png",
    "Chennai": "C:/Users/pavan/Desktop/Capestone_projects/Car_dekho_car_price_prediction/City logos/Chennai.png",
    "Hyderabad": "C:/Users/pavan/Desktop/Capestone_projects/Car_dekho_car_price_prediction/City logos/Hyderabad.png",
    "Kolkata": "C:/Users/pavan/Desktop/Capestone_projects/Car_dekho_car_price_prediction/City logos/Kolkata.png",
    "Jaipur": "C:/Users/pavan/Desktop/Capestone_projects/Car_dekho_car_price_prediction/City logos/Jaipur.png"
}


# Check if a city is selected
if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = None

# Custom CSS for centering and styling the welcome message
st.markdown(
    """
    <style>
    .centered-text {
        text-align: center;
        color: #FF4545;  
    }
    .main-title {
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 24px;
        font-weight: bold;
        margin-top: 0;
    }
    .small-text {
        font-size: 20px;
        color: #FFFFFF; /* Subtle color for instructions */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Welcome page
if st.session_state["selected_city"] is None:
    st.markdown('<p class="centered-text main-title">Welcome!</p>', unsafe_allow_html=True)
    st.markdown('<p class="centered-text sub-title">Predict your used car price</p>', unsafe_allow_html=True)
    st.markdown('<p class="centered-text small-text">choose your city</p>', unsafe_allow_html=True)
    
    # Display cities as clickable options with icons
    col1, col2, col3 = st.columns(3)
    for i, (city, img_path) in enumerate(cities.items()):
        col = [col1, col2, col3][i % 3]
        with col:
            st.image(img_path, width=150)  # Display the image with a fixed width
            if st.button(city, key=city):  # Use the city name as the button label
                st.session_state["selected_city"] = city

else:
    # Prediction Page
    st.markdown('<p class="centered-text sub-title">Please fill car specifiations</p>', unsafe_allow_html=True)

    # Loading the trained model and scaler using pickle
    with open("car_price_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    # loading pickle
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    # Defining categories for one-hot encoding as we have encoded while training our model
    fuel_types = ['petrol', 'diesel', 'electric', 'cng', 'lpg']
    body_types = ['suv', 'Minivans','hatchback', 'sedan','muv', 'hybrids', 'coupe', 'pickup trucks', 'convertibles', 'Wagon']
    transmissions = ['manual', 'automatic']
    insurances = ['third party', 'comprehensive','zero dep','not available']
    turbochargers = ['yes', 'no','twin','turbo']
    tyre_types = ['tubeless radial', 'tubeless', 'run-flat','radial','tubeless runflat']
    manufacturers = ['kia', 'maruti', 'nissan', 'hyundai', 'honda', 'mercedes-benz', 'bmw', 'ford', 'tata', 'jeep', 
                 'toyota', 'audi', 'mahindra', 'renault', 'chevrolet', 'volkswagen', 'datsun', 'fiat', 'land rover',
                 'mg', 'skoda', 'isuzu', 'mini', 'volvo', 'jaguar', 'citroen', 'mitsubishi', 'mahindra renault', 
                 'mahindra ssangyong', 'lexus', 'hindustan motors', 'opel', 'porsche']
    cities = ['chennai', 'hyderabad', 'bangalore', 'delhi', 'jaipur', 'kolkata']


    # Sidebar inputs for categorical features
    st.sidebar.header("Car Specifications")
    manufacturer = st.sidebar.selectbox("Manufacturer", ["Select"] + manufacturers)
    fuel_type = st.sidebar.selectbox("Fuel Type", ["Select"] + fuel_types)
    body_type = st.sidebar.selectbox("Body Type", ["Select"] + body_types)
    transmission = st.sidebar.selectbox("Transmission", ["Select"] + transmissions)
    insurance = st.sidebar.selectbox("Insurance", ["Select"] + insurances)
    turbo_charger = st.sidebar.selectbox("Turbo Charger", ["Select"] + turbochargers)
    tyre_type = st.sidebar.selectbox("Tyre Type", ["Select"] + tyre_types)



    col1, col2, col3 = st.columns(3)

    with col1:
        model_year = st.selectbox("Model Year", ["Select"] + list(range(1990, 2025)))
        engine = st.number_input("Engine Size (cc)", min_value=500.0, max_value=5000.0, step=100.0,value=1400.00)
        gear_box = st.number_input("Gear Box (Speeds)", min_value=3, max_value=10, step=1,value=5)
        height = st.number_input("Height (mm)", min_value=1000.0, max_value=3000.0, step=10.0,value=1500.00)
    

    with col2:
        kilometers_driven = st.number_input("Kilometers Driven")
        torque = st.number_input("Torque (Nm)", min_value=50.0, max_value=1000.0, step=10.0,value=180.00) 
        no_of_cylinders = st.number_input("Number of Cylinders", min_value=2, max_value=12, step=1,value=4)  
        cargo_volumn = st.number_input("Cargo Volumn (L)", min_value=0.0, max_value=1000.0, step=10.0,value=350.00)
    

    with col3:
        owner_no = st.selectbox("Number of Owners", ["Select"]+[1, 2, 3, 4])
        max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=500.0, step=10.0,value=105.0)
        wheel_size = st.number_input("Wheel Size (inches)", min_value=10.0, max_value=30.0, step=1.0,value=15.75)
     


    # Preparing input data for prediction
    input_data = {
        "fuel_type": fuel_type if fuel_type != "Select" else None,
        "body_type": body_type if body_type != "Select" else None,
        "kilometers_driven": kilometers_driven,
        "transmission": transmission if transmission != "Select" else None,
        "owner_no": owner_no if owner_no != "Select" else None,
        "manufacturer": manufacturer if manufacturer != "Select" else None,
        "model_year": model_year,
        "insurance": insurance if insurance != "Select" else None,
        "engine": engine,
        "max_power": max_power,
        "torque": torque,
        "wheel_size": wheel_size,
        "no_of_cylinders": no_of_cylinders,
        "turbo_charger": turbo_charger if turbo_charger != "Select" else None,
        "height": height,
        "gear_box": gear_box,
        "tyre_type": tyre_type if tyre_type != "Select" else None,
        "cargo_volumn": cargo_volumn,
        "city": st.session_state["selected_city"]
    }

        # Check if all inputs are filled
    if None in input_data.values():
        st.warning("Please fill out all fields before predicting.")
    else:
        # Converting input_data to a DataFrame and apply transformations
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df).reindex(columns=model.feature_names_in_, fill_value=0)
        numerical_cols = [
        'kilometers_driven', 'owner_no', 'model_year', 'engine', 'max_power', 
        'torque', 'wheel_size', 'no_of_cylinders', 'height', 'gear_box', 'cargo_volumn'
        ]
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Predicting and display results
        if st.button("Predict Price"):
            prediction = model.predict(input_df)[0]
        # Displaying the prediction in INR, centered and large
            st.markdown(
                f"<h2 style='text-align: center; color: White;'>Estimated Car Price: ₹{prediction:,.2f} lakhs </h2>",
                unsafe_allow_html=True
            )
