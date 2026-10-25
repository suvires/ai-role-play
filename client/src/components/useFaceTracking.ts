// Finds the user's face in their camera, so the avatar can look them in the eye.
// MediaPipe runs in the browser; model and wasm are served from public/mediapipe/.

import { useEffect, useRef } from "react";
import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";

export type FacePosition = {
  /** -1..1, positive when the user is to their right (screen right as they see it) */
  x: number;
  /** -1..1, positive when the user is higher up */
  y: number;
  /** performance.now() of the detection */
  at: number;
};

const DETECTIONS_PER_SECOND = 10;

/** Latest face position from the given camera track (null until a face is seen). */
export function useFaceTracking(track: MediaStreamTrack | null) {
  const faceRef = useRef<FacePosition | null>(null);

  useEffect(() => {
    if (!track) return;
    let cancelled = false;
    let detector: FaceDetector | null = null;
    let timer = 0;

    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.srcObject = new MediaStream([track]);
    video.play().catch(() => {});

    (async () => {
      const vision = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
      if (cancelled) return;
      detector = await FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: "/mediapipe/blaze_face_short_range.tflite", delegate: "GPU" },
        runningMode: "VIDEO",
      });
      if (cancelled) return detector.close();

      const detect = () => {
        if (detector && video.readyState >= 2 && video.videoWidth > 0) {
          const now = performance.now();
          const box = detector.detectForVideo(video, now).detections[0]?.boundingBox;
          if (box) {
            // The camera faces the user, so their right is the image's left
            const centerX = (box.originX + box.width / 2) / video.videoWidth;
            const centerY = (box.originY + box.height / 2) / video.videoHeight;
            faceRef.current = { x: 1 - 2 * centerX, y: 1 - 2 * centerY, at: now };
          }
        }
        timer = window.setTimeout(detect, 1000 / DETECTIONS_PER_SECOND);
      };
      detect();
    })().catch((error) => console.warn("Face tracking unavailable:", error));

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      detector?.close();
      video.srcObject = null;
      faceRef.current = null;
    };
  }, [track]);

  return faceRef;
}
