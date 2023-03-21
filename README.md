## A scientific AI project made for the CS345 Course in the American College of Thessaloniki
Goal: Review the importance of graph-computed features through a comparison between model predictions.

## Prerequisites to running the project
* Docker
* docker-compose
* Anaconda (or equivalent py environment manager)
* Python 3.9

## How to run this project
1. Make sure your environment is set up properly. You may use the `environment.yml` file attached to this project to inspect dependencies.
2. Use `conda env create -f environment.yml` to create the environment. The environment is named `py39torch`. **Remember to review that you don't have an existing enviroment with the same name prior to creation**.
3. Still in the parent directory, run `docker-compose up -d` to initiate Neo4j.
4. While the container is starting up, navigate to the `setup` folder. You will have to run `./import-fraud-database.bash`. This script will import the Bank Fraud Dataset into Neo4j. When the script exits, you will have to wait for a short amount of time until the database restarts.
5. Now you should be good to go. Run `src/main.py` to get started with the program.

## Debugging
* If you are experiencing an error about dataset ingestion when running the models, try running `docker restart cs345-neo4j`. Since the bash script uses an internal admin command to restart the server, sometimes the restart does not work properly.