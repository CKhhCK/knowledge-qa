"""Auth API routes — register, login, get current user."""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.auth.models import RegisterRequest, LoginRequest, TokenResponse, UserInfo
from app.auth.security import create_token, verify_token
from app.auth.user_store import get_user_store
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])
security = HTTPBearer(auto_error=False)


# --- Endpoints ---

@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    """注册新用户，成功后直接返回 token（自动登录）"""
    store = get_user_store()
    result = store.create_user(req.username, req.password, req.email or "")

    if not result["success"]:
        raise HTTPException(status_code=409, detail=result["message"])

    token = create_token({
        "sub": result["user_id"],
        "username": result["username"],
    })

    logger.info(f"User registered and logged in: {result['username']}")
    return TokenResponse(
        access_token=token,
        username=result["username"],
        user_id=result["user_id"],
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """用户登录，返回 JWT token"""
    store = get_user_store()
    user = store.authenticate(req.username, req.password)

    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = create_token({
        "sub": user["user_id"],
        "username": user["username"],
    })

    logger.info(f"User logged in: {user['username']}")
    return TokenResponse(
        access_token=token,
        username=user["username"],
        user_id=user["user_id"],
    )


@router.get("/me", response_model=UserInfo)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """获取当前登录用户信息"""
    if not credentials:
        raise HTTPException(status_code=401, detail="未登录，请先登录")

    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")

    store = get_user_store()
    user = store.get_user(payload["sub"])
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    return UserInfo(**user)


# --- Helper for other routes ---

def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Dependency: extract user_id from JWT, or return 'anonymous'."""
    if not credentials:
        return "anonymous"
    payload = verify_token(credentials.credentials)
    if not payload:
        return "anonymous"
    return payload.get("sub", "anonymous")
