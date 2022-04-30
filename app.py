
from Detection.fake_detector import Matcher
import streamlit as st
import tkinter
import os
import Denomination.denominationGetter as denominationGetter
import preprocessing.note_cropper as cropper
import Detection.fake_detector as fake_detector
from temp_matcher import runner
from PIL import Image


def main():
    '''Main function'''
    img_file = st.sidebar.file_uploader(
        label='', type=['png', 'jpg', 'jpeg'], help="upload image to be evaluated")
    if img_file:
        save_uploaded_file(img_file)
        st.success("Image Cropped")
        denominationGetter.getDenomination("./tmp/cropped/crpd.png")
        try:
            cropped_img = Image.open("./tmp/cropped/crpd.png")
            st.image(cropped_img, caption="Cropped Image",
                     use_column_width=True)
            st.write("Do you want to use the cropped image of the original image?")
            denomination = None
            path = ""
            img = None
            if st.button("Use Cropped Image"):
                path = "./tmp/cropped/crpd.png"
                denomination, img = denominationGetter.getDenomination(path)

            elif st.button("Use Original Image"):
                path = "./tmp/"+img_file.name
                denomination, img = denominationGetter.getDenomination(path)

            if (denomination == ""):
                st.error("This does not seem to be a vailid image")
            elif(denomination == None):
                pass
            else:
                st.success("Denomination: {}".format(denomination))
                st.image(img, caption="knn matched image")

                # add template matching here
                template_images, template_arr = runner(path, denomination)

                # combine avg value with template matching, if valid print images else say it's fake
                percent_matches = 0
                if template_images != None:
                    for clone, template in template_images:
                        st.image(template, caption="security features")
                        st.image(clone, caption="security feature bounding box")
                    template_avg = sum(template_arr)/len(template_arr)
                    num_matches = sum([1 if x > 0.75 else 0 for x in template_arr])
                    percent_matches = num_matches/len(template_arr)*100
                    print(percent_matches)
                    print(template_avg)
                    if percent_matches > 80:
                        st.success("This seems to be a legit note")
                    else:
                        st.error("This seems to be a fake note")
                else:
                    st.error("This seems to be a fake note")

                if percent_matches < 80:
                    images, avg = fake_detector.Matcher(path, denomination)
                    if images != None:
                        for img in images:
                            st.image(img, caption="security features")
                        st.success("This seems to be a legit note")
                    else:
                        st.error("This seems to be a fake note")

                print("done")

        except Exception as e:
            st.error("Error: {}".format(e))


def save_uploaded_file(uploadedfile):
    '''Saves the selected image in the directory specified'''
    with open(os.path.join("./tmp", uploadedfile.name), "wb+") as f:
        f.write(uploadedfile.getbuffer())

        # if the cropped is messing up, skip the cropping part by commenting this line
        cropper.f("./tmp/"+uploadedfile.name)

        # uncomment this line to directy skip cropping and do detection of input image
        # denominationGetter.getDenomination("files/tmp/" + uploadedfile.name)
    return st.success("Image Uploaded  {}".format(uploadedfile.name))


if __name__ == '__main__':
    main()
