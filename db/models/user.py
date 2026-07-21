from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, text as sa_text
from sqlalchemy.orm import relationship

from db.base import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    faculty_id = Column(
        Integer,
        ForeignKey("faculty.faculty_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    email = Column(String(255), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=True)
    role = Column(String(50), nullable=False, default="normal_user")
    created_at = Column(DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"))
    updated_at = Column(DateTime(timezone=True), server_default=sa_text("CURRENT_TIMESTAMP"))

    faculty = relationship("Faculty")