# ContextCheck

A web application that parses online articles/text and measures bias via a linguistic patterns based machine learning algorithm. 

## How To Run
1) First clone the repo `git clone https://github.com/BostonUniversitySeniorDesign/21-22-newsbias.git`

2) Then start the virtual environment `cd 21-22-newsbias/react-flask-app/ && pipenv shell`

3) Then download the backend dependencies `cd api && pipenv install`

4) Then open a new tab in your terminal (rooted at `/react-flask-app`)

5) Then download frontend dependencies `npm i` (and then `npm audit fix` if prompted)

6) Download `features.ckpt`, `data.zip`, and `lexicons.zip`

7) Create a folder @ `/ML/saved_models` and place `features.ckpt` in there

8) Unzip `data` and `lexicons`, place `lexicons` in `data` and place `data` @ `/ML/data`

9) From the first terminal tab, with the pipenv running, start the flask server `flask run`

10) Lastly, from the second tab, start the react server `yarn start` and the webapp should open in your browser

## ML 

All code related to the bias detection machine learning algorithm.

## react-flask-app

Our web app prototype. Front-end UI is written in ReactJS and back-end that interacts with the ML model is written in Python3 and uses Flask. 

## Documentation

Research from the brainstorming/development process, meeting notes, etc. 





