To run this app you must flow the following steps:

- pip install -r requirements.txt
- flask run (if you don't install Flask, please install it first - pip install Flask)

To Run Test:
"""py -m pytest -v test/WebSocketJobTest.py"""

- python3 -m unittest test_app.py
- py -m unittest ./test/WebSocketJobTest.py
- python -m pytest
  """
  More info:
  pip freeze: list all the packages installed in your virtualenv
  pip uninstall -r requirements.txt : uninstall all the packages in requirements.txt
  pip uninstall -r requirements.txt --yes : uninstall all the packages in requirements.txt and remove the folder
  """

Link site packages Python: c:\users\asus\appdata\local\programs\python\python310\lib\site-packages

/// Use virtualenv
//Step 1: Create a virtualenv
pip install virtualenv

//Step 2: Install the requirements
pip install -r requirements.txt
//Step 3: Run the app
flask run
