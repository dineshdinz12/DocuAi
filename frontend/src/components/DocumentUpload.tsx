import { useState, useRef } from "react";
import { UploadCloud, File, CheckCircle, XCircle } from "lucide-react";

import { getSessionId } from "@/utils/session";

interface DocumentUploadProps {
  onUploadSuccess: () => void;
}

export function DocumentUpload({ onUploadSuccess }: DocumentUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<"idle" | "uploading" | "success" | "error">("idle");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileUpload = async (selectedFile: File) => {
    setFile(selectedFile);
    setUploadStatus("uploading");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "";
      const response = await fetch(`${baseUrl}/api/v1/documents/upload`, {
        method: "POST",
        headers: {
          "x-session-id": getSessionId()
        },
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");
      
      setUploadStatus("success");
      onUploadSuccess();
      
      setTimeout(() => {
        setUploadStatus("idle");
        setFile(null);
      }, 3000);
      
    } catch (error) {
      setUploadStatus("error");
      setTimeout(() => setUploadStatus("idle"), 3000);
    }
  };

  return (
    <div className="w-full">
      <div 
        className={`border-2 border-dashed rounded-xl p-4 text-center transition-colors cursor-pointer
          ${isDragging ? "border-indigo-500 bg-indigo-50" : "border-gray-200 bg-white"}
          ${uploadStatus === "uploading" ? "opacity-50 pointer-events-none" : "hover:border-indigo-400"}
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <UploadCloud className="w-8 h-8 mx-auto text-indigo-500 mb-2" />
        <p className="text-xs font-semibold text-gray-700">Drag & drop document</p>
        <p className="text-[10px] text-gray-500 mt-1">PDF only</p>
        
        <input 
          type="file" 
          ref={fileInputRef} 
          className="hidden" 
          accept=".pdf"
          onChange={(e) => e.target.files && handleFileUpload(e.target.files[0])}
        />
      </div>

      {uploadStatus !== "idle" && file && (
        <div className="mt-3 flex items-center p-2 bg-gray-50 rounded-lg border border-gray-100">
          <File className="w-4 h-4 text-indigo-500 mr-2" />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium text-gray-900 truncate">{file.name}</p>
          </div>
          <div className="ml-2">
            {uploadStatus === "uploading" && <div className="w-4 h-4 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />}
            {uploadStatus === "success" && <CheckCircle className="w-4 h-4 text-green-500" />}
            {uploadStatus === "error" && <XCircle className="w-4 h-4 text-red-500" />}
          </div>
        </div>
      )}
    </div>
  );
}
