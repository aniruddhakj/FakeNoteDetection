from Detection.fake_detector import Matcher
import streamlit as st
import tkinter
import os
import Denomination.denominationGetter as denominationGetter
import preprocessing.note_cropper as cropper
import Detection.fake_detector as fake_detector
from PIL import Image


def main():
    '''Main function'''
    img_file = st.sidebar.file_uploader(
        label='', type=['png', 'jpg', 'jpeg'], help="upload image to be evaluated")
    if img_file:
        save_uploaded_file(img_file)
        st.success("Image Cropped")
        denominationGetter.getDenomination("./temp/cropped/crpd.png")
        try:
            cropped_img = Image.open("./temp/cropped/crpd.png")
            st.image(cropped_img, caption="Cropped Image",
                     use_column_width=True)
            st.write("Do you want to use the cropped image of the original image?")
            denomination = None
            path = ""
            if st.button("Use Cropped Image"):
                path = "./temp/cropped/crpd.png"
                denomination = denominationGetter.getDenomination(path)

            elif st.button("Use Original Image"):
                path = "./temp/"+img_file.name
                denomination = denominationGetter.getDenomination(path)

            if (denomination == ""):
                st.error("This does not seem to be a vailid image")
            elif(denomination == None):
                pass
            else:
                st.success("Denomination: {}".format(denomination))
                fake_detector.Matcher(path, denomination)
                print("done")

        except Exception as e:
            st.error("Error: {}".format(e))


def save_uploaded_file(uploadedfile):
    '''Saves the selected image in the directory specified'''
    with open(os.path.join("./temp", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

        # if the cropped is messing up, skip the cropping part by commenting this line
        cropper.f("./temp/"+uploadedfile.name)

        # uncomment this line to directy skip cropping and do detection of input image
        # denominationGetter.getDenomination("files/temp/" + uploadedfile.name)
    return st.success("Image Uploaded  {}".format(uploadedfile.name))


if __name__ == '__main__':
    main()
