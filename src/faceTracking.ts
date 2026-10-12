import { FaceLandmarker, FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision'

/** Face + upper-body tracking. Pose is optional: face tracking remains available if pose setup fails. */
export interface BodyReadout {
  tilt: number        // -1..1, shoulder-line/torso inclination
  sway: number        // -1..1, upper-body displacement relative to the hips
  shrug: number       // -1..1, shoulder height relative to initial posture
}
export interface FaceReadout {
  blink: number
  mouth: number
  smile: number
  body?: BodyReadout | null
}

const VERSION = '1.1.0'
const WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n))
const clamp01 = (n: number) => clamp(n, 0, 1)
const smooth = (a: number, b: number, alpha: number) => a + (b - a) * alpha

type Point = { x: number; y: number; visibility?: number }

export async function startFaceDetection(
  video: HTMLVideoElement,
  report: (values: FaceReadout | null) => void,
): Promise<() => void> {
  const resolver = await FilesetResolver.forVisionTasks(WASM)
  const face = await FaceLandmarker.createFromOptions(resolver, {
    baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'CPU' },
    runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true,
  })

  let stopped = false
  let pose: PoseLandmarker | null = null
  let timer: number | undefined
  let lastVideoTime = -1
  let lastPoseTime = 0
  let lastBody: BodyReadout | null = null
  let baseline: { height: number; slope: number } | null = null
  const started = performance.now()

  // Load the lighter pose model separately. Do not break already-working face tracking.
  void PoseLandmarker.createFromOptions(resolver, {
    baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'CPU' },
    runningMode: 'VIDEO', numPoses: 1,
  }).then(instance => {
    if (stopped) instance.close()
    else pose = instance
  }).catch(error => console.warn('[FaceTrack] Pose unavailable; continuing with face tracking:', error))

  const run = () => {
    if (stopped) return
    if (video.readyState >= 2 && video.videoWidth > 0 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime
      const now = performance.now()
      try {
        const result = face.detectForVideo(video, now)
        const categories = result.faceBlendshapes?.[0]?.categories
        if (!categories) {
          report(null)
        } else {
          const score = (name: string) => categories.find(c => c.categoryName === name)?.score ?? 0
          // Pose runs at most ~10fps so iPhone can render 4096px Live2D meanwhile.
          if (pose && now - lastPoseTime >= 100) {
            lastPoseTime = now
            try {
              const points = pose.detectForVideo(video, now).landmarks?.[0] as Point[] | undefined
              const l = points?.[11], r = points?.[12], lh = points?.[23], rh = points?.[24]
              if (l && r && (l.visibility ?? 1) > .45 && (r.visibility ?? 1) > .45) {
                const shoulderWidth = Math.max(.045, Math.hypot(r.x - l.x, r.y - l.y))
                const centerX = (l.x + r.x) / 2
                const centerY = (l.y + r.y) / 2
                const slope = Math.atan2(r.y - l.y, r.x - l.x)
                const hipsVisible = !!(lh && rh && (lh.visibility ?? 1) > .35 && (rh.visibility ?? 1) > .35)
                const hipX = hipsVisible ? (lh!.x + rh!.x) / 2 : .5
                const hipY = hipsVisible ? (lh!.y + rh!.y) / 2 : centerY + shoulderWidth * 1.35
                const torsoHeight = Math.max(.04, hipY - centerY)
                // Body movement is camera-relative, not inferred from facial expression.
                const signedShift = (centerX - hipX) / shoulderWidth
                if (!baseline) baseline = { height: torsoHeight / shoulderWidth, slope }
                const raw: BodyReadout = {
                  tilt: clamp((slope - baseline.slope) / .38, -1, 1),
                  sway: clamp(signedShift / .5, -1, 1),
                  shrug: clamp(((torsoHeight / shoulderWidth) - baseline.height) / .40, -1, 1),
                }
                lastBody = lastBody ? {
                  tilt: smooth(lastBody.tilt, raw.tilt, .32),
                  sway: smooth(lastBody.sway, raw.sway, .32),
                  shrug: smooth(lastBody.shrug, raw.shrug, .25),
                } : raw
              } else {
                lastBody = null
              }
            } catch (error) {
              console.warn('[FaceTrack] Pose frame failed:', error)
            }
          }
          report({
            blink: clamp01((score('eyeBlinkLeft') + score('eyeBlinkRight')) / 2),
            mouth: clamp01(score('jawOpen')),
            smile: clamp01((score('mouthSmileLeft') + score('mouthSmileRight')) / 2),
            body: lastBody,
          })
        }
      } catch (error) { console.warn('[FaceTrack] Face frame failed:', error) }
    }
    // Approximately 15fps detection. Rendering still uses requestAnimationFrame.
    timer = window.setTimeout(run, 67)
  }
  run()
  return () => {
    stopped = true
    if (timer !== undefined) clearTimeout(timer)
    face.close()
    pose?.close()
  }
}
