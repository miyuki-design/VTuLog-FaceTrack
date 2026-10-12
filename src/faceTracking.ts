import { FaceLandmarker, FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision'

/** Tracked inputs. All values are normalized to -1..1 or 0..1. */
export interface BodyReadout { tilt: number; sway: number; shrug: number }
export interface FaceReadout {
  blink: number; mouth: number; smile: number
  leftBlink?: number; rightBlink?: number
  leftWide?: number; rightWide?: number
  headX?: number; headY?: number; headZ?: number
  eyeX?: number; eyeY?: number
  body?: BodyReadout | null
}
const VERSION = '1.1.0'
const WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n))
const lerp = (a: number, b: number, alpha: number) => a + (b - a) * alpha
const radToDegrees = 180 / Math.PI

type Point = { x: number; y: number; visibility?: number }

export async function startFaceDetection(
  video: HTMLVideoElement,
  report: (values: FaceReadout | null) => void,
): Promise<() => void> {
  const resolver = await FilesetResolver.forVisionTasks(WASM)
  const face = await FaceLandmarker.createFromOptions(resolver, {
    baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'CPU' },
    runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  })
  let stopped = false
  let pose: PoseLandmarker | null = null
  let timer: number | undefined
  let lastVideoTime = -1
  let lastPoseTime = 0
  let lastBody: BodyReadout | null = null
  let baselineBody: { height: number; slope: number; shift: number } | null = null
  let baselineHead: { x: number; y: number; z: number } | null = null
  let lastHead = { x: 0, y: 0, z: 0 }
  // Pose is optional: keep the working face detector even if pose initialization fails.
  void PoseLandmarker.createFromOptions(resolver, {
    baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'CPU' },
    runningMode: 'VIDEO', numPoses: 1,
  }).then(instance => { if (stopped) instance.close(); else pose = instance })
    .catch(error => console.warn('[FaceTrack] Pose unavailable:', error))

  const run = () => {
    if (stopped) return
    if (video.readyState >= 2 && video.videoWidth > 0 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime
      const now = performance.now()
      try {
        const result = face.detectForVideo(video, now)
        const categories = result.faceBlendshapes?.[0]?.categories
        if (!categories) {
          baselineHead = null
          lastBody = null
          baselineBody = null
          report(null)
        } else {
          const score = (name: string) => categories.find(c => c.categoryName === name)?.score ?? 0
          const matrix = result.facialTransformationMatrixes?.[0]?.data
          if (matrix && matrix.length >= 16) {
            // MediaPipe's facial transform is a row-major 4x4 matrix.
            const raw = {
              x: Math.atan2(matrix[8], matrix[10]) * radToDegrees,
              y: Math.atan2(-matrix[9], Math.hypot(matrix[8], matrix[10])) * radToDegrees,
              z: Math.atan2(matrix[4], matrix[5]) * radToDegrees,
            }
            if (!baselineHead) baselineHead = raw
            const target = {
              x: clamp((raw.x - baselineHead.x) / 25, -1, 1),
              y: clamp((raw.y - baselineHead.y) / 20, -1, 1),
              z: clamp((raw.z - baselineHead.z) / 25, -1, 1),
            }
            lastHead = {
              x: lerp(lastHead.x, target.x, .45),
              y: lerp(lastHead.y, target.y, .45),
              z: lerp(lastHead.z, target.z, .45),
            }
          }
          if (pose && now - lastPoseTime >= 100) {
            lastPoseTime = now
            try {
              const pts = pose.detectForVideo(video, now).landmarks?.[0] as Point[] | undefined
              const l = pts?.[11], r = pts?.[12], lh = pts?.[23], rh = pts?.[24]
              if (l && r && (l.visibility ?? 1) > .45 && (r.visibility ?? 1) > .45) {
                const width = Math.max(.045, Math.hypot(r.x - l.x, r.y - l.y))
                const cx = (l.x + r.x) / 2, cy = (l.y + r.y) / 2
                const slope = Math.atan2(r.y - l.y, r.x - l.x)
                const hipsVisible = !!(lh && rh && (lh.visibility ?? 1) > .35 && (rh.visibility ?? 1) > .35)
                const hx = hipsVisible ? (lh!.x + rh!.x) / 2 : .5
                const hy = hipsVisible ? (lh!.y + rh!.y) / 2 : cy + width * 1.35
                const h = Math.max(.04, hy - cy) / width
                const shift = (cx - hx) / width
                if (!baselineBody) baselineBody = { height: h, slope, shift }
                const raw: BodyReadout = {
                  tilt: clamp((slope - baselineBody.slope) / .38, -1, 1),
                  sway: clamp((shift - baselineBody.shift) / .5, -1, 1),
                  shrug: clamp((h - baselineBody.height) / .4, -1, 1),
                }
                lastBody = lastBody ? {
                  tilt: lerp(lastBody.tilt, raw.tilt, .32),
                  sway: lerp(lastBody.sway, raw.sway, .32),
                  shrug: lerp(lastBody.shrug, raw.shrug, .25),
                } : raw
              } else lastBody = null
            } catch (error) { console.warn('[FaceTrack] Pose frame failed:', error) }
          }
          const lb = score('eyeBlinkLeft'), rb = score('eyeBlinkRight')
          // Eye look categories give relative direction, not exact gaze angles.
          report({
            blink: (lb + rb) / 2,
            mouth: score('jawOpen'), smile: (score('mouthSmileLeft') + score('mouthSmileRight')) / 2,
            leftBlink: lb, rightBlink: rb,
            leftWide: score('eyeWideLeft'), rightWide: score('eyeWideRight'),
            headX: lastHead.x, headY: lastHead.y, headZ: lastHead.z,
            eyeX: clamp((score('eyeLookOutRight') + score('eyeLookInLeft') - score('eyeLookInRight') - score('eyeLookOutLeft')), -1, 1),
            eyeY: clamp((score('eyeLookUpLeft') + score('eyeLookUpRight') - score('eyeLookDownLeft') - score('eyeLookDownRight')), -1, 1),
            body: lastBody,
          })
        }
      } catch (error) { console.warn('[FaceTrack] Face frame failed:', error) }
    }
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
