import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

/** Stage 1 — local face detection only. No changes to Live2D or its 4096 canvas. */
export interface FaceReadout { blink: number; mouth: number; smile: number }
type FaceCategory = { categoryName: string; score: number }
type FaceResult = { faceBlendshapes?: Array<{ categories: FaceCategory[] }> }
type FaceLandmarkerInstance = {
  detectForVideo: (video: HTMLVideoElement, timestampMs: number) => FaceResult
  close: () => void
}

const VERSION = '0.10.22'
const WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`
const MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
const clamp01 = (n: number) => Math.max(0, Math.min(1, n))

export async function startFaceDetection(
  video: HTMLVideoElement,
  report: (values: FaceReadout | null) => void,
): Promise<() => void> {
  // Vite leaves this dynamic CDN URL untouched. No lockfile / package change needed.
  const resolver = await visionModule.FilesetResolver.forVisionTasks(WASM)
  const landmarker = await visionModule.FaceLandmarker.createFromOptions(resolver, {
    baseOptions: { modelAssetPath: MODEL, delegate: 'CPU' },
    runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true,
  })
  let stopped = false
  let timeout: number | undefined
  let lastTime = -1
  const run = () => {
    if (stopped) return
    if (video.readyState >= 2 && video.videoWidth > 0 && video.currentTime !== lastTime) {
      lastTime = video.currentTime
      try {
        const result = landmarker.detectForVideo(video, performance.now())
        const scores = result.faceBlendshapes?.[0]?.categories
        if (!scores) report(null)
        else {
          const score = (name: string) => scores.find(s => s.categoryName === name)?.score ?? 0
          report({
            blink: clamp01((score('eyeBlinkLeft') + score('eyeBlinkRight')) / 2),
            mouth: clamp01(score('jawOpen')),
            smile: clamp01((score('mouthSmileLeft') + score('mouthSmileRight')) / 2),
          })
        }
      } catch (error) { console.warn('Face detection error:', error) }
    }
    // Run face detection at ~8fps to keep CPU/memory headroom on iPhones.
    timeout = window.setTimeout(run, 125)
  }
  run()
  return () => {
    stopped = true
    if (timeout !== undefined) clearTimeout(timeout)
    landmarker.close()
  }
}
