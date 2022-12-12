import streamlit
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.header("HATE THIS")
