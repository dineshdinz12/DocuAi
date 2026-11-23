import boto3
from botocore.config import Config
from fastapi import UploadFile
import uuid
import os
import re
from datetime import datetime
from core.config import settings

class DocumentService:
    def __init__(self):
        self.use_s3 = False
        self.local_dir = "/tmp/docuai_storage"
        os.makedirs(self.local_dir, exist_ok=True)
        
        try:
            endpoint = f"http://{settings.MINIO_ENDPOINT}" if not settings.MINIO_ENDPOINT.startswith("http") else settings.MINIO_ENDPOINT
            self.s3_client = boto3.client(
                's3',
                endpoint_url=endpoint,
                aws_access_key_id=settings.MINIO_ACCESS_KEY,
                aws_secret_access_key=settings.MINIO_SECRET_KEY,
                config=Config(s3={"addressing_style": "path"}),
            )
            self.bucket = settings.MINIO_BUCKET
            self.s3_client.head_bucket(Bucket=self.bucket)
            self.use_s3 = True
            print("Connected to MinIO/S3 object storage.")
        except Exception as e:
            if settings.APP_ENV == "production":
                raise RuntimeError("Object storage is unavailable") from e
            print(f"MinIO/S3 not available ({e}). Falling back to local disk storage at {self.local_dir}")
            self.use_s3 = False

    def upload_document(self, file: UploadFile, session_id: str = "default_session") -> tuple:
        doc_id = str(uuid.uuid4())
        safe_filename = re.sub(r"[^A-Za-z0-9._ -]", "_", os.path.basename(file.filename or "document.pdf"))
        
        if self.use_s3:
            object_name = f"{session_id}/{doc_id}_{safe_filename}"
            temp_path = f"/tmp/{doc_id}_{safe_filename}"
            with open(temp_path, "wb") as buffer:
                buffer.write(file.file.read())
                
            self.s3_client.upload_file(
                temp_path, 
                self.bucket, 
                object_name,
                ExtraArgs={
                    "Metadata": {
                        "original_filename": safe_filename,
                        "session_id": session_id
                    }
                }
            )
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return doc_id, safe_filename, f"s3://{self.bucket}/{object_name}"
        else:
            session_dir = os.path.join(self.local_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)
            file_path = os.path.join(session_dir, f"{doc_id}_{safe_filename}")
            with open(file_path, "wb") as buffer:
                buffer.write(file.file.read())
            return doc_id, safe_filename, file_path

    def download_to_path(self, storage_key: str, destination: str) -> str:
        if self.use_s3 and storage_key.startswith(f"s3://{self.bucket}/"):
            object_name = storage_key.removeprefix(f"s3://{self.bucket}/")
            self.s3_client.download_file(self.bucket, object_name, destination)
            return destination
        return storage_key

    def list_documents(self, session_id: str = "default_session"):
        if self.use_s3:
            try:
                prefix = f"{session_id}/"
                response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
                docs = []
                if 'Contents' in response:
                    for obj in response['Contents']:
                        key = obj['Key']
                        try:
                            head = self.s3_client.head_object(Bucket=self.bucket, Key=key)
                            original_filename = head.get('Metadata', {}).get('original_filename', key.split('/')[-1])
                        except Exception:
                            original_filename = key.split('/')[-1]
                            
                        raw_filename = key.split('/')[-1]
                        doc_id = raw_filename.split('_')[0] if '_' in raw_filename else raw_filename
                        docs.append({
                            "id": doc_id,
                            "name": original_filename,
                            "key": key,
                            "size": obj['Size'],
                            "uploaded_at": obj['LastModified'].isoformat()
                        })
                docs.sort(key=lambda x: x['uploaded_at'], reverse=True)
                return docs
            except Exception as e:
                print(f"Error listing documents for session {session_id}: {e}")
                return []
        else:
            session_dir = os.path.join(self.local_dir, session_id)
            if not os.path.exists(session_dir):
                return []
            docs = []
            for fname in os.listdir(session_dir):
                full_path = os.path.join(session_dir, fname)
                if os.path.isfile(full_path):
                    parts = fname.split('_', 1)
                    doc_id = parts[0]
                    orig_name = parts[1] if len(parts) > 1 else fname
                    stat = os.stat(full_path)
                    docs.append({
                        "id": doc_id,
                        "name": orig_name,
                        "key": full_path,
                        "size": stat.st_size,
                        "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
            docs.sort(key=lambda x: x['uploaded_at'], reverse=True)
            return docs

    def delete_document(self, doc_id: str, session_id: str = "default_session"):
        docs = self.list_documents(session_id=session_id)
        doc_to_delete = next((d for d in docs if d["id"] == doc_id), None)
        if not doc_to_delete:
            return False
        key = doc_to_delete["key"]
        return self.delete_stored_file(f"s3://{self.bucket}/{key}" if self.use_s3 else key)

    def delete_stored_file(self, storage_key: str) -> bool:
        prefix = f"s3://{self.bucket}/"
        try:
            if storage_key.startswith("s3://"):
                if not self.use_s3 or not storage_key.startswith(prefix):
                    return False
                self.s3_client.delete_object(Bucket=self.bucket, Key=storage_key.removeprefix(prefix))
            else:
                try:
                    os.unlink(storage_key)
                except FileNotFoundError:
                    pass
            return True
        except Exception as error:
            print(f"Document storage deletion failed: {type(error).__name__}")
            return False

    def get_presigned_url(self, doc_id: str, session_id: str = "default_session"):
        docs = self.list_documents(session_id=session_id)
        doc = next((d for d in docs if d["id"] == doc_id), None)
        if not doc:
            return None
            
        if self.use_s3:
            try:
                return self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.bucket, 'Key': doc["key"]},
                    ExpiresIn=3600
                )
            except Exception:
                return None
        else:
            return f"/api/v1/documents/{doc_id}/download?session_id={session_id}"

document_service = DocumentService()
