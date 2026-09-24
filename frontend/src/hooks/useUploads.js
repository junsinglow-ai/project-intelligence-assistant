import { useCallback, useRef, useState } from "react";
import { MAX_UPLOAD_MB, UPLOAD_SUFFIXES, uploadFile } from "../api/client.js";
import { newId } from "../lib/storage.js";

// How long a finished upload's chip stays in the composer before it clears.
const DONE_CHIP_MS = 4000;

function rejectReason(file) {
  const dot = file.name.lastIndexOf(".");
  const suffix = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
  if (!UPLOAD_SUFFIXES.includes(suffix)) {
    return `Unsupported file type ${suffix || "(none)"}; use ${UPLOAD_SUFFIXES.join(", ")}`;
  }
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `File exceeds ${MAX_UPLOAD_MB}MB`;
  return null;
}

// Upload state shared by the composer's paperclip and the chat drop zone.
export function useUploads(onUploaded) {
  const [attachments, setAttachments] = useState([]);
  // The callback changes every render of the parent; read it through a ref so
  // an in-flight upload reports to the current one.
  const onUploadedRef = useRef(onUploaded);
  onUploadedRef.current = onUploaded;

  const patch = useCallback((id, fields) => {
    setAttachments((list) => list.map((a) => (a.id === id ? { ...a, ...fields } : a)));
  }, []);

  const dismiss = useCallback((id) => {
    setAttachments((list) => list.filter((a) => a.id !== id));
  }, []);

  const addFiles = useCallback(
    (files) => {
      for (const file of Array.from(files)) {
        const id = newId();
        const reason = rejectReason(file);
        setAttachments((list) => [
          ...list,
          {
            id,
            name: file.name,
            status: reason ? "error" : "uploading",
            progress: 0,
            error: reason,
          },
        ]);
        if (reason) continue;

        uploadFile(file, (progress) => patch(id, { progress }))
          .then((result) => {
            patch(id, { status: "done", progress: 1, result });
            onUploadedRef.current?.(result);
            setTimeout(() => dismiss(id), DONE_CHIP_MS);
          })
          .catch((err) => patch(id, { status: "error", error: err.message }));
      }
    },
    [patch, dismiss]
  );

  const uploading = attachments.some((a) => a.status === "uploading");
  return { attachments, addFiles, dismiss, uploading };
}
