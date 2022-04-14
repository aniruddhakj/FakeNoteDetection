import streamlit as st
import os
import denominationGetter as denominationGetter
from note_cropper import noteCrop
#import temp_matcher
import fake_detector

def main():
    '''Main function'''
    img_file = st.sidebar.file_uploader(
        label='', type=['png', 'jpg', 'jpeg'], help="upload image to be evaluated")
    if img_file:
        _, denom, path = save_uploaded_file(img_file) 
        os.mkdir("./working_dir")
        noteCrop(path)
        path = "./working_dir/note.png"
        #temp_matcher.runner(path, denom)
        fake_detector.ORBrunner(path, denom)
        fake_detector.SIFTrunner(path,denom)
        os.rmdir("./working_dir")

    

def save_uploaded_file(uploadedfile):
    '''Saves the selected image in the directory specified'''
    with open(os.path.join("./files/Test", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
        denom = denominationGetter.getDenomination("files/Test/" + uploadedfile.name)
    return st.success("Selected image {}".format(uploadedfile.name)), denom, "files/Test/" + uploadedfile.name


if __name__ == '__main__':
    main()