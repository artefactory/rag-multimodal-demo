from pathlib import Path

from fastapi import Depends, HTTPException, Response, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt

from backend import ADMIN_MODE
from backend.api_plugins.lib.user_management import (
    ALGORITHM,
    SECRET_KEY,
    UnsecureUser,
    User,
    authenticate_user,
    create_access_token,
    create_user,
    delete_user,
    get_user,
    user_exists,
)


def authentication_routes(app, dependencies=list[Depends]):
    from backend.database import Database

    with Database() as connection:
        connection.run_script(Path(__file__).parent / "users_tables.sql")

    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/login")

    async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email: str = payload.get("email")
            if email is None:
                raise credentials_exception

            user = get_user(email)
            if user is None:
                raise credentials_exception
            return user
        except JWTError:
            raise credentials_exception

    @app.post("/user/signup", include_in_schema=ADMIN_MODE)
    async def signup(user: UnsecureUser) -> dict:
        if not ADMIN_MODE:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Signup is disabled"
            )

        user = User.from_unsecure_user(user)
        if user_exists(user.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"User {user.email} already registered",
            )

        create_user(user)
        return {"email": user.email}

    @app.delete("/user/")
    async def del_user(current_user: User = Depends(get_current_user)) -> dict:
        email = current_user.email
        try:
            user = get_user(email)
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User {email} not found",
                )
            delete_user(email)
            return {"detail": f"User {email} deleted"}
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal Server Error",
            )

    @app.post("/user/login")
    async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> dict:
        user = authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user_data = user.dict()
        del user_data["hashed_password"]
        access_token = create_access_token(data=user_data)
        return {"access_token": access_token, "token_type": "bearer"}

    @app.get("/user/me")
    async def user_me(current_user: User = Depends(get_current_user)) -> User:
        return current_user

    @app.get("/user")
    async def user_root() -> dict:
        return Response("User management routes are enabled.", status_code=200)

    return Depends(get_current_user)
