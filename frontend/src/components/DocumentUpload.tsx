import { useState, useRef } from "react";
import { Paperclip, FileText, CheckCircle2, AlertCircle, Loader2, Plus, UploadCloud } from "lucide-react";
import { getSessionId } from "@/utils/session";

interface DocumentUploadProps {
  onUploadSuccess: () => void;
  variant?: "button" | "icon";
}

export function DocumentUpload({ onUploadSuccess, variant = "button" }: DocumentUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<"idle" | "uploading" | "success" | "error">("idle");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileUpload = async (selectedFile: File) => {
    if (!selectedFile.name.endsWith(".pdf")) {
      setUploadStatus("error");
      setTimeout(() => setUploadStatus("idle"), 3000);
      return;
    }

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

  if (variant === "icon") {
    return (
      <div className="relative inline-flex items-center">
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="p-2 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-xl transition-all"
          title="Upload PDF Document"
          disabled={uploadStatus === "uploading"}
        >
          {uploadStatus === "uploading" ? (
            <Loader2 className="w-4 h-4 text-slate-700 animate-spin" />
          ) : (
            <Paperclip className="w-4 h-4" strokeWidth={1.5} />
          )}
        </button>

        <input 
          type="file" 
          ref={fileInputRef} 
          className="hidden" 
          accept=".pdf"
          onChange={(e) => e.target.files && handleFileUpload(e.target.files[0])}
        />
      </div>
    );
  }

  return (
    <div className="w-full">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`w-full p-5 border border-dashed rounded-xl cursor-pointer text-center transition-all duration-200 group relative ${
          isDragging
            ? "border-slate-900 bg-slate-100/50"
            : "border-slate-200/80 hover:border-slate-400 bg-white hover:bg-slate-50/50"
        } ${uploadStatus === "uploading" ? "opacity-50 pointer-events-none" : ""}`}
      >
        {uploadStatus === "uploading" ? (
          <div className="flex flex-col items-center justify-center py-2">
            <Loader2 className="w-6 h-6 text-slate-800 animate-spin mb-2" />
            <span className="text-xs font-semibold text-slate-700">Uploading PDF...</span>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center">
            <div className="p-2 bg-slate-100 rounded-lg mb-2 text-slate-500 group-hover:text-slate-800 transition-colors">
              <UploadCloud className="w-5 h-5" strokeWidth={1.5} />
            </div>
            <p className="text-xs font-bold text-slate-800">Drag files here</p>
            <p className="text-[10px] text-slate-400 mt-1 font-semibold">or click to upload from device</p>
          </div>
        )}
      </div>

      <input 
        type="file" 
        ref={fileInputRef} 
        className="hidden" 
        accept=".pdf"
        onChange={(e) => e.target.files && handleFileUpload(e.target.files[0])}
      />

      {uploadStatus !== "idle" && file && (
        <div className="mt-2 flex items-center p-2 bg-white rounded-lg border border-slate-200 shadow-sm animate-in fade-in duration-200">
          <FileText className="w-3.5 h-3.5 text-slate-400 mr-2 flex-shrink-0" strokeWidth={1.5} />
          <div className="flex-1 min-w-0">
            <p className="text-[11px] font-medium text-slate-800 truncate">{file.name}</p>
          </div>
          <div className="ml-1">
            {uploadStatus === "success" && (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" strokeWidth={1.75} />
            )}
            {uploadStatus === "error" && (
              <AlertCircle className="w-3.5 h-3.5 text-rose-500" strokeWidth={1.75} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
