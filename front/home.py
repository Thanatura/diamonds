import streamlit as st

st.title("Diamonds Price Prediction App")

st.markdown("""This is a simple Streamlit application that allows you to predict the price of a diamond based on its characteristics. 
            You can input the features of the diamond and get an estimated price using a machine learning model deployed as an API.""")

# Logo Artefact
st.markdown("<img src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAAP1BMVEVHcEwKESgKESgRESsKESgOESoTES0XEi4fEjItEzkhEzNDEUEhEzRjDUppDEyGBlSNB1cLESgXES4KESgKESiXRhehAAAAFXRSTlMANcsn/8T/wv//xP/C/5EIAr+GtXOG2sbdAAAAtUlEQVR4AX3LURZEQAwF0dARCQD2v9Z5DUB66rfOpUdZRunyPP0Cc0jOgrn4A9O0EJSgQdZ8Wupa6UJVUzNVj1ZmVhuqHIgVw//SBq/tuha38WDb9kPbOrSJsyMCtTcNGmdPNMSpT1rGCQgaZ/mAomo1IGhtqnKno4jaRGtTnOMNMqYOtIUnHC6IKYA7FczxgoxLZ3jM4YQI8KSMxhMiusWx4MCLevCiDryoCy86M1rec2E0/wCazgih6xhvCQAAAABJRU5ErkJggg==' alt='Logo' style='width:200px;'>", unsafe_allow_html=True)