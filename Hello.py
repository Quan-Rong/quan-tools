# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Hello",
    page_icon="👋", layout="wide"
)

st.write('''# The Tools''')

main_image=Image.open('logo_main_11.JPG')
st.image(main_image, caption='Version: Beta V1.0')

st.sidebar.success("Select a locacase above.")

st.markdown('''---''')

st.markdown(
    """
    * Complete APP deployment, local/cloud. Docker installation runs on Linux.
    * Optimize K&C result analysis.
    * Integrate Static Loads Generation feature.
    * Integrate KnC database.
    * Integrate suspension computer.
"""
)

st.markdown('''---''')

def main():
    cs_body()
    
def cs_body():

    col1, col2 = st.columns(2)

    with col1:
        main_image_01=Image.open('logo_main_12.JPG')    
        st.image(main_image_01)
        st.write("✈️K&C Simulation Results PostProcess")
    
        main_image_02=Image.open('logo_main_13.JPG')    
        st.image(main_image_02)
        st.write("🤸🏻Gestamp Static Loads")

# 在第二个列中显示上下两张图片，并在每张图片下面写一行字
    with col2:
        main_image_03=Image.open('logo_main_14.JPG')    
        st.image(main_image_03)
        st.write("👨🏻‍💻K&C DataBase Analysis")
    
        main_image_04=Image.open('logo_main_15.JPG')    
        st.image(main_image_04)
        st.write("🕵🏻Suspension Parameter Calculation")

    return None

# Run main()

if __name__ == '__main__':
    main()
