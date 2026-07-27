import asyncio
from services.document_service import document_service
from fastapi import UploadFile
import io

async def test():
    class DummyFile:
        def read(self):
            return b"dummy content"
            
    class DummyUpload:
        filename = "test.pdf"
        file = DummyFile()
        
    print("Uploading...")
    res = document_service.upload_document(DummyUpload())
    print("Result:", res)

asyncio.run(test())
