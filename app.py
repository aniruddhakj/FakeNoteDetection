import streamlit as st
import os
import denominationGetter


def main():
    '''Main function'''
    img_file = st.sidebar.file_uploader(
        label='', type=['png', 'jpg', 'jpeg'], help="upload image to be evaluated")
    if img_file:
        save_uploaded_file(img_file)


def save_uploaded_file(uploadedfile):
    '''Saves the selected image in the directory specified'''
    with open(os.path.join("./files/Test", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
        denominationGetter.getDenomination("files/Test/" + uploadedfile.name)
    return st.success("Selected image {}".format(uploadedfile.name))


if __name__ == '__main__':
    main()