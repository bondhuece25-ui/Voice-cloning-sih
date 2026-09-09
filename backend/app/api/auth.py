from fastapi import APIRouter
from pydantic import BaseModel

from app.db.supabase import supabase

router = APIRouter()


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/login")
def login(data: LoginRequest):

    response = supabase.auth.sign_in_with_password({
        "email": data.email,
        "password": data.password
    })

    return {
        "access_token": response.session.access_token,
        "user_id": str(response.user.id)
    }