# API Food Recommendation
Deploy the FastAPI Machine Learning model using Cloud Run

## Contributor
|            Name          |  Bangkit ID  |       Path       |
|:------------------------:|:------------:|:----------------:|
|  Danica Kirana           |  C306DSY0893 | Cloud Computing  |
|  Jeremy Kevin A. S.      |  C306DSX1475 | Cloud Computing  |

## Requirements
* Pyenv (optional)
* Virtualenv (optional)
* Python 3.9
* Google Cloud Platform Account
* Google Cloud Platform - Cloud Build API


## Running Locally
```
$ virtualenv -p python3.8.2 .venv
$ source .venv/bin/activate
$ pip3 install -r requirements.txt
$ uvicorn main:app --reload
```

## Deploying to Cloud Run using Cloud SDK
```
$ gcloud init
$ gcloud services enable run.googleapis.com
$ gcloud builds submit --tag gcr.io/[your-project-id]/[your-folder]
After that, create service using Google Clod Console
```

## Deploying to Cloud Run using Google Cloud Platform

1. Make sure you have an active Google Cloud Platform (GCP) account. If not, sign up and create a new project at https://console.cloud.google.com.

2. Ensure you have installed the Google Cloud SDK (https://cloud.google.com/sdk) and initialized it by running the following command in the terminal or command prompt:
   ``` gcloud init ```

3. Create a repository on a code management service like GitHub or GitLab, and make sure the repository contains all the necessary files for your FastAPI application, including Dockerfile, requirements.txt, and your FastAPI application code.

4. Open a terminal or command prompt and navigate to the directory where you want to clone the FastAPI repository.

5. Clone the FastAPI repository by running the following command:
   ` git clone https://github.com/danicakirana/C23PS423_CC `

6. After the cloning process is complete, navigate your terminal or command prompt to the newly cloned FastAPI directory.

7. Build the local Docker container by running the following command:
   ` docker build -t gcr.io/[PROJECT_ID]/[your-folder] . `
   Replace [PROJECT_ID] with your designated Google Cloud Platform project ID.

8. Once the building process is complete, verify that the local Docker container is running by executing the following command:
   ` docker run -p 8080:8080 gcr.io/[PROJECT_ID]/[your-folder] `
   Make sure there are no errors and that the FastAPI application runs properly on localhost.

9. If the previous step is successful, stop and remove the running Docker container by pressing Ctrl+C in the terminal or command prompt.

10. To publish the Docker container to the Google Cloud Container Registry, execute the following command:
    ` docker push gcr.io/[PROJECT_ID]/[your-folder] `
    The container will be uploaded to the Container Registry in the corresponding Google Cloud Platform project.

11. Next, create a Cloud Run service in Google Cloud console.

12. After the deployment process is complete, GCP will provide a URL that can be used to access the deployed FastAPI application. Copy that URL from the output and try accessing it in a web browser or using an API testing tool like Postman.

## Demo of FastAPI Models
To try the demo of the above model, you can open the following link: https://rekomendasi-uctmtl3fka-et.a.run.app/docs
You can use the example input with UserId: A3SGXH7AUHU8GW
