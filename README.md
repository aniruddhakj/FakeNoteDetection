# IPCV_Project
A project aimed at identifying fake Indian currency notes

## Index
- [Getting Started](https://github.com/aniruddhakj/IPCV_Project/blob/main/README.md#getting-started)
- [Running the App](https://github.com/aniruddhakj/IPCV_project/blob/main/README.md#running-the-app)

## Getting Started
1. Download and install Python3 from [this link](https://www.python.org/downloads/)
2. Install [venv](https://pypi.org/project/virtualenv/) to create a virtual environment for the project.
    - For Windows, You can do this using a terminal and type :
        ```bash
        py -m pip install --user virtualenv
        ```
    - For macOS and Linux:
        ```zsh
        python3 -m pip install --user virtualenv
        ```  
3. Now create and activate the virtual environment
    ```bash
    venv IPCV
    ```
    - For Windows
        ```bash
        cd IPCV\Scripts
        activate
        ```
    - For macOS and Linux
      ```zsh
      source IPCV/bin/activate
      ```
4. Clone the repo, install the required modules using requirements.txt
     ```zsh
    pip3 install -r requirements.txt
    ```
    
## Running the App
* Run the following command in the terminal to run the app
     ```zsh
    streamlit run app.py
    ```
