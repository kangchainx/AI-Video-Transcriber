import asyncio
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

try:
    from minio import Minio
    from minio.error import S3Error
except Exception:  # pragma: no cover - minio is optional
    Minio = None
    S3Error = Exception

logger = logging.getLogger(__name__)


class MinioStorage:
    """Encapsulates MinIO upload logic behind an optional interface."""

    def __init__(self, client: Optional["Minio"], bucket: Optional[str], prefix: str = ""):
        self._client = client
        self._bucket = bucket
        self._prefix = prefix.strip("/") if prefix else ""
        self._bucket_checked = False

    @property
    def enabled(self) -> bool:
        return self._client is not None and self._bucket is not None

    @classmethod
    def from_env(cls) -> "MinioStorage":
        endpoint = os.getenv("MINIO_ENDPOINT")
        access_key = os.getenv("MINIO_ACCESS_KEY")
        secret_key = os.getenv("MINIO_SECRET_KEY")
        bucket = os.getenv("MINIO_BUCKET")
        secure = os.getenv("MINIO_SECURE", "true").lower() == "true"
        region = os.getenv("MINIO_REGION")
        prefix = os.getenv("MINIO_PREFIX", "")

        if Minio is None:
            logger.info("[MinIO] 未安装minio依赖，跳过云端上传")
            return cls(None, None)

        if not all([endpoint, access_key, secret_key, bucket]):
            logger.info("[MinIO] 环境变量缺失，上传功能禁用")
            return cls(None, None)

        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region,
        )
        logger.info(
            "[MinIO] 客户端已配置 endpoint=%s secure=%s bucket=%s prefix=%s",
            endpoint,
            secure,
            bucket,
            prefix or "<none>"
        )
        return cls(client, bucket, prefix)

    def _object_name(self, object_name: str) -> str:
        cleaned = object_name.lstrip("/")
        if self._prefix:
            return f"{self._prefix}/{cleaned}"
        return cleaned

    async def _ensure_bucket(self) -> None:
        if self._bucket_checked or not self.enabled:
            return

        async def _check_and_create():
            exists = await asyncio.to_thread(self._client.bucket_exists, self._bucket)
            if not exists:
                logger.info("[MinIO] 桶 %s 不存在，正在创建", self._bucket)
                await asyncio.to_thread(self._client.make_bucket, self._bucket)
                logger.info("[MinIO] 桶 %s 创建完成", self._bucket)
            else:
                logger.debug("[MinIO] 桶 %s 已存在", self._bucket)

        try:
            await _check_and_create()
            self._bucket_checked = True
        except S3Error as exc:
            logger.error("[MinIO] 桶校验失败: %s", exc)
            raise

    async def upload_file(self, local_path: Path, object_name: str) -> Optional[str]:
        """上传本地文件到MinIO，返回对象键。"""
        if not self.enabled:
            return None

        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        await self._ensure_bucket()
        obj = self._object_name(object_name)

        try:
            logger.info("[MinIO] 上传开始 path=%s object=%s", path, obj)
            await asyncio.to_thread(
                self._client.fput_object,
                self._bucket,
                obj,
                str(path),
            )
            logger.info("[MinIO] 上传成功 path=%s object=%s", path, obj)
            return obj
        except Exception as exc:
            logger.error("[MinIO] 上传失败 (%s -> %s): %s", path, obj, exc)
            raise

    async def get_presigned_url(self, object_name: Optional[str], expires: Union[int, float, timedelta] = 3600) -> Optional[str]:
        if not self.enabled or not object_name:
            return None
        expiry = expires
        if isinstance(expiry, (int, float)):
            expiry = timedelta(seconds=float(expiry))
        try:
            url = await asyncio.to_thread(
                self._client.presigned_get_object,
                self._bucket,
                object_name,
                expiry,
            )
            logger.debug("[MinIO] 生成下载链接 object=%s", object_name)
            return url
        except Exception as exc:
            logger.error("[MinIO] 生成下载链接失败 (%s): %s", object_name, exc)
            return None

    async def stat_object(self, object_name: Optional[str]):
        if not self.enabled or not object_name:
            return None
        try:
            stat = await asyncio.to_thread(
                self._client.stat_object,
                self._bucket,
                object_name,
            )
            logger.debug("[MinIO] 获取对象信息成功 object=%s", object_name)
            return stat
        except Exception as exc:
            logger.error("[MinIO] 获取对象信息失败 (%s): %s", object_name, exc)
            return None
