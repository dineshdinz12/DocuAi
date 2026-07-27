import boto3
from fastapi import UploadFile
import uuid
import os
from core.config import settings

class DocumentService:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=f"http://{settings.MINIO_ENDPOINT}",
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
        )
        self.bucket = settings.MINIO_BUCKET
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.s3_client.create_bucket(Bucket=self.bucket)

    def upload_document(self, file: UploadFile, session_id: str = "default_session") -> tuple:
        doc_id = str(uuid.uuid4())
        object_name = f"{session_id}/{doc_id}_{file.filename}"
        
        # Save temp file
        temp_path = f"/tmp/{doc_id}_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(file.file.read())
            
        # Upload to MinIO with metadata
        self.s3_client.upload_file(
            temp_path, 
            self.bucket, 
            object_name,
            ExtraArgs={
                "Metadata": {
                    "original_filename": file.filename,
                    "session_id": session_id
                }
            }
        )
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return doc_id, file.filename, f"s3://{self.bucket}/{object_name}"

    def list_documents(self, session_id: str = "default_session"):
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
            # Sort by newest
            docs.sort(key=lambda x: x['uploaded_at'], reverse=True)
            return docs
        except Exception as e:
            print(f"Error listing documents for session {session_id}: {e}")
            return []

    def delete_document(self, doc_id: str, session_id: str = "default_session"):
        try:
            docs = self.list_documents(session_id=session_id)
            doc_to_delete = next((d for d in docs if d["id"] == doc_id), None)
            if doc_to_delete:
                self.s3_client.delete_object(Bucket=self.bucket, Key=doc_to_delete["key"])
                return True
            return False
        except Exception as e:
            print(f"Error deleting document {doc_id}: {e}")
            return False

    def get_presigned_url(self, doc_id: str, session_id: str = "default_session"):
        try:
            docs = self.list_documents(session_id=session_id)
            doc = next((d for d in docs if d["id"] == doc_id), None)
            if doc:
                url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.bucket, 'Key': doc["key"]},
                    ExpiresIn=3600
                )
                return url
            return None
        except Exception as e:
            return None

document_service = DocumentService()
