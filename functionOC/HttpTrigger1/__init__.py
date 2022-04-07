import logging

import azure.functions as func
import json

def main(req: func.HttpRequest) -> func.HttpResponse:
    
    user_id = req.get_json()["userId"]

    recommend = get_recommendations(user_id)

    resp = json.dumps(recommend)

    return func.HttpResponse(resp)


def get_recommendations(user_id):
    return ['0','1','2','3','4']
