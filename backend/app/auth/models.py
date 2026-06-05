"""Auth-related Pydantic models."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class RegisterRequest(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=2, max_length=32, description="用户名（支持中英文）")
    password: str = Field(..., min_length=6, max_length=128, description="密码")
    email: Optional[str] = Field(default=None, max_length=128, description="邮箱（可选）")

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("用户名至少2个字符")
        if len(v) > 32:
            raise ValueError("用户名最长32个字符")
        # Deny pure spaces or control characters
        if any(ord(c) < 32 for c in v):
            raise ValueError("用户名包含非法字符")
        return v


class LoginRequest(BaseModel):
    """User login request."""
    username: str = Field(..., min_length=1, description="用户名")
    password: str = Field(..., min_length=1, description="密码")


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    username: str
    user_id: str


class UserInfo(BaseModel):
    """Public user information."""
    user_id: str
    username: str
    email: Optional[str] = None
    created_at: str
