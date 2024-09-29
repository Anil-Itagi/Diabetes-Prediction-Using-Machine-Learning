import streamlit as st
a=st.text_input("enter the name")
st.button("Test")
st.header("Your salary is around {}".format(a))
print(a)