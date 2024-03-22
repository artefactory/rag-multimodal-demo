from datetime import datetime, timedelta

import argon2
from jose import jwt
from pydantic import BaseModel

from backend import ALGORITHM, SECRET_KEY
from backend.database import Database


class UnsecureUser(BaseModel):
    email: str = None
    password: bytes = None


class User(BaseModel):
    email: str = None
    hashed_password: str = None

    @classmethod
    def from_unsecure_user(cls, unsecure_user: UnsecureUser):
        hashed_password = argon2.hash_password(unsecure_user.password).decode("utf-8")
        return cls(email=unsecure_user.email, hashed_password=hashed_password)


def create_user(user: User) -> None:
    with Database() as connection:
        connection.execute(
            "INSERT INTO users (email, password) VALUES (?, ?)",
            (user.email, user.hashed_password),
        )


def user_exists(email: str) -> bool:
    with Database() as connection:
        result = connection.fetchone("SELECT 1 FROM users WHERE email = ?", (email,))
        return bool(result)


def get_user(email: str) -> User | None:
    with Database() as connection:
        user_row = connection.fetchone("SELECT * FROM users WHERE email = ?", (email,))
        if user_row:
            return User(email=user_row[0], hashed_password=user_row[1])
        return None


def delete_user(email: str) -> None:
    with Database() as connection:
        connection.execute("DELETE FROM users WHERE email = ?", (email,))


def authenticate_user(username: str, password: bytes) -> bool | User:
    user = get_user(username)
    if not user:
        return False

    if argon2.verify_password(
        user.hashed_password.encode("utf-8"), password.encode("utf-8")
    ):
        return user

    return False


def create_access_token(*, data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=60)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
