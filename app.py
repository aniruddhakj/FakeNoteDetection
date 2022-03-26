import streamlit as st
import os
import project

def main():
    img_file = st.sidebar.file_uploader(label='', type=['png', 'jpg'], help="upload image to be evaluated")
    if img_file:
        save_uploaded_file(img_file)

def save_uploaded_file(uploadedfile):
    '''Saves the selected image in Present working directory for google API processing'''
    with open(os.path.join("./files/Test", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
        project.matcher("files/Test/" + uploadedfile.name)
    return st.success("Selected image {}".format(uploadedfile.name))

if __name__ == '__main__':
    main()