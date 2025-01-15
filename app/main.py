from fastapi import FastAPI
from http import HTTPStatus
from enum import Enum
from pydantic import BaseModel
from fastapi import UploadFile, File, Query
from fastapi.responses import FileResponse
import re
from typing import Optional
import cv2

app = FastAPI()

# ------ ROOT URL EXAMPLES ------ #

# When the user visits the root URL, the response will be a JSON object
# with the key "Hello" and the value "World".
#@app.get("/")
#def read_root():
#    return {"Hello": "World"}

# More complex example:
@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response



# ------ GET EXAMPLES ------ #

# When the user visits the URL /items/{item_id}, the response will be a JSON object
# with the key "item_id" and the value of the item_id parameter.
#@app.get("/items/{item_id}")
#def read_item(item_id: int):
#    return {"item_id": item_id}


# Specific path parameters:
class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"
@app.get("/restric_items/{item_id}")
def read_item(item_id: ItemEnum):
    return {"item_id": item_id}


# Query parameters:
@app.get("/query_items")
def read_item_q(item_id: int):
    return {"item_id": item_id}


# ------ POST EXAMPLES ------ #
database = {'username': [ ], 'password': [ ]}

@app.post("/login/")
def login(username: str, password: str):
    username_db = database['username']
    password_db = database['password']
    if username not in username_db and password not in password_db:
        with open('database.csv', "a") as file:
            file.write(f"{username}, {password} \n")
        username_db.append(username)
        password_db.append(password)
    return "login saved"


# Define the input schema
class EmailDomainCheck(BaseModel):
    email: str
    domain_match: str

@app.post("/text_model/")
def contains_email(data: EmailDomainCheck):
    # Regex for email validation
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # Extract domain from the email
    match = re.fullmatch(regex, data.email)
    if not match:
        return {
            "input": data.email,
            "message": "Invalid email format",
            "status-code": HTTPStatus.BAD_REQUEST,
            "is_email": False,
            "domain_match": False,
        }

    # Check if the domain matches
    domain = data.email.split('@')[-1]
    domain_match = domain.startswith(data.domain_match)

    response = {
        "input": data.email,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": True,
        "domain_match": domain_match,
    }
    return response


@app.post("/cv_model/")
async def cv_model(
    data: UploadFile = File(...),
    h: int = Query(28, description="Height of the resized image"),
    w: int = Query(28, description="Width of the resized image"),
):
    # Save the uploaded file
    with open("image.jpg", "wb") as image:
        content = await data.read()
        image.write(content)

    # Read and resize the image
    img = cv2.imread("image.jpg")
    if img is None:
        return {
            "message": "Uploaded file is not a valid image",
            "status-code": HTTPStatus.BAD_REQUEST,
        }

    res = cv2.resize(img, (w, h))

    # Save the resized image
    resized_image_path = "image_resize.jpg"
    cv2.imwrite(resized_image_path, res)

    # Return the resized image as a file response
    return FileResponse(resized_image_path, media_type="image/jpeg", filename="resized_image.jpg")



# to call use: uvicorn --reload --port 8000 main:app
# then go to http://localhost:8000/docs in your browser
