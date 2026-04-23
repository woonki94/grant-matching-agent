from typing import Optional
from sqlalchemy.orm import Session

from db.models.user import User


class UserDAO:
    def __init__(self, session: Session):
        self.session = session

    def get_by_email(self, email: str) -> Optional[User]:
        if not email:
            return None
        return (
            self.session.query(User)
            .filter(User.email == email)
            .one_or_none()
        )

    def create_user(
        self,
        faculty_id: int,
        email: str,
        password_hash: str,
        role: str = "normal_user",
    ) -> User:
        user = User(
            faculty_id=faculty_id,
            email=email,
            password_hash=password_hash,
            role=role,
        )
        self.session.add(user)
        return user

    def set_password(self, user: User, password_hash: str) -> User:
        user.password_hash = password_hash
        return user