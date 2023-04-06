import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from api import logic, utils


def main():
    st.header("Liver Segmentation from Abdominal T1Dual Image")
    model = logic.load_model()
    col_1, col_2 = st.columns(2)
    in_phase_file = col_1.file_uploader("In Phase DICOM", type=["dcm"])
    out_phase_file = col_2.file_uploader("Out Phase DICOM", type=["dcm"])

    if in_phase_file and out_phase_file:
        in_phase_dcm_byte = in_phase_file.read()
        out_phase_dcm_byte = out_phase_file.read()

        in_phase_img = logic.load_dicom_image(in_phase_dcm_byte)
        out_phase_img = logic.load_dicom_image(out_phase_dcm_byte)

        in_phase_img_norm = logic.normalize_image(in_phase_img)
        out_phase_img_norm = logic.normalize_image(out_phase_img)

        model_input = [[in_phase_img_norm, out_phase_img_norm]]

        model_prediction = model.predict(model_input)

        cleaned_model_prediction = (model_prediction[0] > 0.5) * 1

        fig, ax = plt.subplots(1, 3)
        ax[0].set_title("In Phase")
        ax[0].imshow(in_phase_img[0].numpy(), cmap="gray", interpolation="none")
        ax[1].set_title("Out Phase")
        ax[1].imshow(out_phase_img[0].numpy(), cmap="gray", interpolation="none")
        ax[2].set_title("Prediction")
        ax[2].imshow(in_phase_img[0].numpy(), cmap="gray", interpolation="none")
        ax[2].imshow(model_prediction[0], alpha=0.5)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
