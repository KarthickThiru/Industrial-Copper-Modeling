import streamlit as st
import pickle
import numpy as np
import sklearn
from streamlit_option_menu import option_menu
from datetime import date
from streamlit_extras.add_vertical_space import add_vertical_space

#Functions for  "Predict_status()"

def predict_status(country,item_type,application,width,product_reference,quantity_tons,customer_log,thickness_log,selling_price_log,Day_of_item_date,Month_of_Item_date,Year_of_Item_Date,Day_of_Delivery_date,Month_of_Delivery_date,Year_of_Delivery_date):

    # change the datatypes "string" to "int"
    itdd = int(Day_of_item_date)
    itdm = int(Month_of_Item_date)
    itdy = int(Year_of_Item_Date)

    dydd = int(Day_of_Delivery_date)
    dydm = int(Month_of_Delivery_date)
    dydy = int(Year_of_Delivery_date)

    # modelfile of the classification
    with open("K:/DS/industrial_copper_modeling/Classification_model.pkl","rb") as f:

        model_class=pickle.load(f)

    user_data = np.array([[country,item_type,application,width,product_reference,quantity_tons,customer_log,thickness_log,
                       selling_price_log,itdd,itdm,itdy,dydd,dydm,dydy]])
    
    y_pred= model_class.predict(user_data)

    if y_pred == 1:

        return 1

    else:

        return 0

#Functions for  "Predict_selling_price()"

def predict_selling_price(country,sts,item_type,application,width,product_reference,quantity_tons,customer_log,
                   thickness_log,Day_of_item_date,Month_of_Item_date,Year_of_Item_Date,Day_of_Delivery_date,Month_of_Delivery_date,Year_of_Delivery_date):

    # change the datatypes "string" to "int"
    itdd = int(Day_of_item_date)
    itdm = int(Month_of_Item_date)
    itdy = int(Year_of_Item_Date)

    dydd = int(Day_of_Delivery_date)
    dydm = int(Month_of_Delivery_date)
    dydy = int(Year_of_Delivery_date)

    # modelfile of the classification
    with open("K:/DS/industrial_copper_modeling/Regression_Model.pkl","rb") as f:

        model_regg=pickle.load(f)

    user_data= np.array([[country,sts,item_type,application,width,product_reference,quantity_tons,customer_log,thickness_log,
                       itdd,itdm,itdy,dydd,dydm,dydy]])
    
    y_pred= model_regg.predict(user_data)

    ac_y_pred= np.exp(y_pred[0])

    return ac_y_pred

#Streamlit UI Part

st.set_page_config(layout="wide")

st.title("Industrial Copper Modeling 🪽 ")
st.write(" 🧑‍💻 Tech Used: Python scripting, Data Preprocessing, EDA, Streamlit")

select = option_menu(
    menu_title=None,
    options=["About ICM", "Predicting Selling Price", "Predict Status","Conclusion"],
    icons=["grid-1x2", "currency-rupee", "trophy","check-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

if select == 'About ICM':

    b1, b2 = st.columns(2)
    with b1:
        st.write('## **Problem Statement**')
        st.write('* The copper industry deals with less complex data related to sales and pricing.Where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer')
        st.write('* ML Regression model which predicts continuous variable :violet[**‘Selling_Price’**].')
        st.write('* ML Classification model which predicts Status: :green[**WON**] or :red[**LOST**].')
        st.write('## Tools and Technologies used')
        st.write('Python, Streamlit, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Pickle, Streamlit-Option-Menu')
        

    with b2:
        st.write('## **USING MACHINE LEARNING**')

        #st.write("### ML MODELS USED")
        st.write('#### REGRESSION - ***:red[RandomForestRegressor]***')
        st.write('- Random Forest Regression is a versatile machine-learning technique for predicting numerical values. It combines the predictions of multiple decision trees to reduce overfitting and improve accuracy. Python’s machine-learning libraries make it easy to implement and optimize this approach.')
        st.write('#### CLASSIFICATION - ***:violet[RandomForestClassification]***')
        st.write('- Random forest is a commonly-used machine learning algorithm trademarked by Leo Breiman and Adele Cutler, which combines the output of multiple decision trees to reach a single result. Its ease of use and flexibility have fueled its adoption.')

if select == 'Conclusion': 
    
    st.write('## Price and Status of copper is depend on many of the following :')
    st.write("* Copper is often considered a barometer of economic health. Economic growth tends to increase demand for copper in construction, manufacturing, and infrastructure projects.")     
    st.write('* The growing :red[**adoption of electric vehicles (EVs)**] and renewable energy technologies, such as :red[**solar**] and :red[**wind**], can influence copper demand due to its use in wiring and components.')      
    st.write('* Technological advancements that increase the efficiency of copper usage or lead to the development of new applications can impact demand.')
    st.write('* Keep an eye on :violet[**trade policies, tariffs**], and :violet[**geopolitical events**] that may impact the flow of copper between countries. Changes in global trade dynamics can affect prices.  ')
    st.write('* Utilize advanced :orange[**predictive modeling techniques**], such as :orange[**machine learning algorithms**], to analyze historical data and identify patterns that may assist in forecasting future copper trends. ')

if select == "Predict Status":

    st.header(":red[**PREDICT STATUS (Won / Lose)**]")
    st.write(" ")

    col1,col2= st.columns(2)

    with col1:

        country = st.number_input(label="**Enter the Value for COUNTRY**/ Min:25.0, Max:113.0")
        item_type = st.number_input(label="**Enter the Value for ITEM TYPE**/ Min:0.0, Max:6.0")
        application = st.number_input(label="**Enter the Value for APPLICATION**/ Min:2.0, Max:87.5")
        width = st.number_input(label="**Enter the Value for WIDTH**/ Min:700.0, Max:1980.0")
        product_ref = st.number_input(label="**Enter the Value for PRODUCT_REF**/ Min:611728, Max:1722207579")
        quantity_tons_log = st.number_input(label="**Enter the Value for QUANTITY_TONS (Log Value)**/ Min:-0.322, Max:6.924",format="%0.15f")
        customer_log = st.number_input(label="**Enter the Value for CUSTOMER (Log Value)**/ Min:17.21910, Max:17.23015",format="%0.15f")
        thickness_log = st.number_input(label="**Enter the Value for THICKNESS (Log Value)**/ Min:-1.71479, Max:3.28154",format="%0.15f")
    
    with col2:

        selling_price_log = st.number_input(label="**Enter the Value for SELLING PRICE (Log Value)**/ Min:5.97503, Max:7.39036",format="%0.15f")
        item_date_day = st.selectbox("**Select the Day for ITEM DATE**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
        item_date_month = st.selectbox("**Select the Month for ITEM DATE**",("1","2","3","4","5","6","7","8","9","10","11","12"))
        item_date_year = st.selectbox("**Select the Year for ITEM DATE**",("2020","2021"))
        delivery_date_day = st.selectbox("**Select the Day for DELIVERY DATE**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
        delivery_date_month = st.selectbox("**Select the Month for DELIVERY DATE**",("1","2","3","4","5","6","7","8","9","10","11","12"))
        delivery_date_year = st.selectbox("**Select the Year for DELIVERY DATE**",("2020","2021","2022"))
        

    button = st.button(":violet[***PREDICT THE STATUS***]",use_container_width=True)

    if button:

        status = predict_status(country, item_type, application, width, product_ref, quantity_tons_log,
                               customer_log, thickness_log, selling_price_log, item_date_day,
                               item_date_month, item_date_year, delivery_date_day, delivery_date_month,
                               delivery_date_year)
        
        if status == 1:

            st.write("## :green[**The Status is WON**]")

        else:

            st.write("## :red[**The Status is LOSE**]")

if select == "Predicting Selling Price":

    st.header("**PREDICT SELLING PRICE**")
    st.write(" ")

    col1,col2= st.columns(2)

    with col1:

        country = st.number_input(label="**Enter the Value for COUNTRY**/ Min:25.0, Max:113.0")
        status = st.number_input(label="**Enter the Value for STATUS**/ Min:0.0, Max:8.0")
        item_type = st.number_input(label="**Enter the Value for ITEM TYPE**/ Min:0.0, Max:6.0")
        application = st.number_input(label="**Enter the Value for APPLICATION**/ Min:2.0, Max:87.5")
        width = st.number_input(label="**Enter the Value for WIDTH**/ Min:700.0, Max:1980.0")
        product_ref = st.number_input(label="**Enter the Value for PRODUCT_REF**/ Min:611728, Max:1722207579")
        quantity_tons_log = st.number_input(label="**Enter the Value for QUANTITY_TONS (Log Value)**/ Min:-0.3223343801166147, Max:6.924734324081348",format="%0.15f")
        customer_log = st.number_input(label="**Enter the Value for CUSTOMER (Log Value)**/ Min:17.21910565821408, Max:17.230155364880137",format="%0.15f")
        
    
    with col2:

        thickness_log = st.number_input(label="**Enter the Value for THICKNESS (Log Value)**/ Min:-1.7147984280919266, Max:3.281543137578373",format="%0.15f")
        item_date_day = st.selectbox("**Select the Day for ITEM DATE**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
        item_date_month = st.selectbox("**Select the Month for ITEM DATE**",("1","2","3","4","5","6","7","8","9","10","11","12"))
        item_date_year = st.selectbox("**Select the Year for ITEM DATE**",("2020","2021"))
        delivery_date_day = st.selectbox("**Select the Day for DELIVERY DATE**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
        delivery_date_month = st.selectbox("**Select the Month for DELIVERY DATE**",("1","2","3","4","5","6","7","8","9","10","11","12"))
        delivery_date_year = st.selectbox("**Select the Year for DELIVERY DATE**",("2020","2021","2022"))
        

    button= st.button(":violet[***PREDICT THE SELLING PRICE***]",use_container_width=True)

    if button:

        price= predict_selling_price(country,status, item_type, application, width, product_ref, quantity_tons_log,
                               customer_log, thickness_log, item_date_day,
                               item_date_month, item_date_year, delivery_date_day, delivery_date_month,
                               delivery_date_year)
        
        
        st.write("## :green[**The Selling Price is :**]",price)
