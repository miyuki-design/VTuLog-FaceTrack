import { useRef, useState, useCallback, useEffect } from 'react'
import { saveAvatarFiles, loadAvatarFiles } from './avatarStorage'
import type { FaceReadout } from './faceTracking'

export const LIVE2D_CANVAS_SIZE = 4096

// ──────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────

export interface NormalizedFileInfo {
  name: string
  originalPath: string   // webkitRelativePath before normalization (may be empty)
  normalizedPath: string // webkitRelativePath after normalization
}

export interface Live2DDiagnostics {
  fileCount: number
  model3Name: string | null
  mocFile: string | null
  textureCount: number
  physicsFile: string | null
  // Path resolution
  settingsUrl: string | null
  resolvedMocPath: string | null       // raw  resolveURL(moc)
  encodedMocPath: string | null        // encodeURI(resolveURL(moc))
  mocNormalizedPath: string | null     // clone's raw webkitRelativePath
  encodedMocWebkit: string | null      // encodeURI(clone.webkitRelativePath)
  mocPathMatch: boolean | null         // encodedMocPath === encodedMocWebkit
  // Per-file normalization log
  normalizedFiles: NormalizedFileInfo[]
  // NetworkError
  networkErrorUrl?: string
  networkErrorStatus?: number | null
  networkErrorAborted?: boolean
  networkErrorStack?: string
}

export type Live2DStatus = 'idle' | 'loading' | 'loaded' | 'error'

const EMPTY_DIAG: Live2DDiagnostics = {
  fileCount: 0,
  model3Name: null,
  mocFile: null,
  textureCount: 0,
  physicsFile: null,
  settingsUrl: null,
  resolvedMocPath: null,
  encodedMocPath: null,
  mocNormalizedPath: null,
  encodedMocWebkit: null,
  mocPathMatch: null,
  normalizedFiles: [],
}

// ──────────────────────────────────────────────
// Cubism Core loader
// ──────────────────────────────────────────────
async function loadCubismCore(): Promise<void> {
  if ((window as any).Live2DCubismCore) return
  const tryScript = (url: string) =>
    new Promise<void>((resolve, reject) => {
      const s = document.createElement('script')
      s.src = url
      s.onload = () => resolve()
      s.onerror = () => { s.remove(); reject() }
      document.head.appendChild(s)
    })
  try {
    await tryScript('https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js')
  } catch {
    try {
      await tryScript('/live2dcubismcore.min.js')
    } catch {
      throw new Error(
        'Cubism Core が読み込めませんでした。\n' +
        'Live2D 公式サイトから SDK をダウンロードし、\n' +
        'live2dcubismcore.min.js を /public/ に配置してください。'
      )
    }
  }
}

// ──────────────────────────────────────────────
// Clone a File with a forced webkitRelativePath
// ──────────────────────────────────────────────
function cloneWithPath(file: File, path: string): File {
  const clone = new File([file], file.name, {
    type: file.type,
    lastModified: file.lastModified,
  })
  Object.defineProperty(clone, 'webkitRelativePath', { value: path, writable: false })
  return clone
}

// ──────────────────────────────────────────────
// Collect every referenced relative path from model3.json FileReferences
// ──────────────────────────────────────────────
function collectExpectedPaths(fr: any): string[] {
  const paths: string[] = []
  const add = (p: unknown) => { if (typeof p === 'string' && p) paths.push(p) }

  add(fr.Moc)
  add(fr.Physics)
  add(fr.Pose)
  add(fr.UserData)
  if (Array.isArray(fr.Textures))    fr.Textures.forEach(add)
  if (Array.isArray(fr.Expressions)) fr.Expressions.forEach((e: any) => add(e?.File))
  if (fr.Motions && typeof fr.Motions === 'object') {
    for (const group of Object.values(fr.Motions) as any[][]) {
      if (Array.isArray(group)) group.forEach((m: any) => { add(m?.File); add(m?.Sound) })
    }
  }
  return paths
}

// ──────────────────────────────────────────────
// Normalize files:
//   • model3.json  → webkitRelativePath = model3File.name  (bare filename)
//   • every other  → webkitRelativePath = the relative path written in model3.json
// This makes settings.resolveURL(rel) === file.webkitRelativePath for all refs.
// ──────────────────────────────────────────────
interface NormalizeResult {
  normalizedFiles: File[]
  log: NormalizedFileInfo[]
}

function normalizeFiles(files: File[], model3File: File, fr: any): NormalizeResult {
  const expectedPaths = collectExpectedPaths(fr)

  // Build basename → expected path map (first occurrence wins)
  const basenameToExpected = new Map<string, string>()
  for (const ep of expectedPaths) {
    const base = ep.split('/').pop()!
    if (!basenameToExpected.has(base)) basenameToExpected.set(base, ep)
  }

  const log: NormalizedFileInfo[] = []
  const normalizedFiles: File[] = []

  for (const file of files) {
    const originalPath = file.webkitRelativePath || ''

    if (file === model3File) {
      // model3.json: bare filename so settings.url = name, resolveURL works from root
      const np = file.name
      normalizedFiles.push(cloneWithPath(file, np))
      log.push({ name: file.name, originalPath, normalizedPath: np })
      continue
    }

    const expectedPath = basenameToExpected.get(file.name)
    if (expectedPath) {
      normalizedFiles.push(cloneWithPath(file, expectedPath))
      log.push({ name: file.name, originalPath, normalizedPath: expectedPath })
    } else {
      // Not referenced — keep original path or bare name, include anyway
      const np = originalPath || file.name
      normalizedFiles.push(cloneWithPath(file, np))
      log.push({ name: file.name, originalPath, normalizedPath: np })
    }
  }

  return { normalizedFiles, log }
}

// ──────────────────────────────────────────────
// NetworkError decoder
// ──────────────────────────────────────────────
function parseNetworkError(e: unknown): {
  message: string; url?: string; status?: number | null
  aborted?: boolean; stack?: string
} {
  if (!e || typeof e !== 'object') return { message: String(e) }
  const err = e as any
  return {
    message: err.message ?? 'Network error',
    url:     err.url     ?? undefined,
    status:  typeof err.status  === 'number'  ? err.status  : undefined,
    aborted: typeof err.aborted === 'boolean' ? err.aborted : undefined,
    stack:   typeof err.stack   === 'string'  ? err.stack   : undefined,
  }
}

// ──────────────────────────────────────────────
// Hook
// ──────────────────────────────────────────────
export function useLive2D() {
  const pixiAppRef           = useRef<any>(null)
  const pixiCanvasRef        = useRef<HTMLCanvasElement | null>(null)
  const loadedRef            = useRef(false)
  const faceReadoutRef       = useRef<FaceReadout | null>(null)
  const settingsObjectURLRef = useRef<string | null>(null)

  const [status,      setStatus]      = useState<Live2DStatus>('idle')
  const [errorMsg,    setErrorMsg]    = useState<string | null>(null)
  const [modelName,   setModelName]   = useState('')
  const [diagnostics, setDiagnostics] = useState<Live2DDiagnostics | null>(null)

  const cleanup = useCallback(() => {
    if (pixiAppRef.current) {
      try { pixiAppRef.current.destroy(true) } catch { /* ignore */ }
      pixiAppRef.current = null
    }
    if (settingsObjectURLRef.current) {
      try { URL.revokeObjectURL(settingsObjectURLRef.current) } catch { /* ignore */ }
      settingsObjectURLRef.current = null
    }
    pixiCanvasRef.current = null
    faceReadoutRef.current = null
    loadedRef.current = false
    setStatus('idle')
    setErrorMsg(null)
    setDiagnostics(null)
  }, [])

  const loadModel = useCallback(async (files: File[]) => {
    cleanup()
    setStatus('loading')

    // Initial quick scan
    const model3File = files.find(f => f.name.endsWith('.model3.json')) ?? null
    const initialDiag: Live2DDiagnostics = {
      ...EMPTY_DIAG,
      fileCount:    files.length,
      model3Name:   model3File?.name ?? null,
      mocFile:      files.find(f => f.name.endsWith('.moc3'))?.name ?? null,
      textureCount: files.filter(f => /\.(png|jpg|jpeg|webp)$/i.test(f.name)).length,
      physicsFile:  files.find(f => f.name.endsWith('physics3.json'))?.name ?? null,
    }
    setDiagnostics(initialDiag)
    setModelName(initialDiag.model3Name?.replace('.model3.json', '') ?? 'モデル')

    if (!model3File) {
      setErrorMsg('.model3.json が見つかりません（フォルダごと選択されましたか？）')
      setStatus('error')
      return
    }

    try {
      // ── Parse model3.json ──
      let json: any
      try {
        json = JSON.parse(await model3File.text())
      } catch {
        throw new Error(`${model3File.name} の JSON 解析に失敗しました`)
      }
      const fr: any = json.FileReferences ?? {}

      // ── Normalize all files ──
      // iPhone Safari may have empty webkitRelativePath; clone each file with
      // the path that model3.json expects so FileLoader's validateFiles() finds them.
      const { normalizedFiles, log } = normalizeFiles(files, model3File, fr)

      // ── Build Cubism4ModelSettings ──
      // json.url = model3File.name (bare) so resolveURL("Foo.moc3") === "Foo.moc3",
      // which now matches the normalised webkitRelativePath of the moc3 clone.
      json.url = model3File.name

      await loadCubismCore()

      const PIXI = await import('pixi.js')
      ;(window as any).PIXI = PIXI
      await new Promise<void>(r => setTimeout(r, 0))

      const { Live2DModel, Cubism4ModelSettings } = await import('pixi-live2d-display/cubism4')

      const settings = new (Cubism4ModelSettings as any)(json)
      const settingsObjectURL = URL.createObjectURL(model3File)
      settings._objectURL = settingsObjectURL

      // ── Compute raw and encoded moc paths for comparison ──
      const tryResolve = (p: string) => { try { return settings.resolveURL(p) } catch { return null } }
      const resolvedMocPath: string | null = settings.moc ? tryResolve(settings.moc) : null
      const encodedMocPath:  string | null = resolvedMocPath ? encodeURI(resolvedMocPath) : null

      const mocClone = normalizedFiles.find(f => f.name.endsWith('.moc3'))
      const mocNormalizedPath: string | null = mocClone?.webkitRelativePath ?? null
      const encodedMocWebkit:  string | null = mocNormalizedPath ? encodeURI(mocNormalizedPath) : null

      // FileLoader compares encodeURI(file.webkitRelativePath) with encodeURI(resolveURL(path))
      const mocPathMatch =
        encodedMocPath !== null && encodedMocWebkit !== null
          ? encodedMocPath === encodedMocWebkit
          : null

      setDiagnostics({
        ...initialDiag,
        settingsUrl:       settings.url ?? json.url,
        resolvedMocPath,
        encodedMocPath,
        mocNormalizedPath,
        encodedMocWebkit,
        mocPathMatch,
        normalizedFiles:   log,
      })

      // Guard before attempting load
      if (mocPathMatch === false) {
        throw new Error(
          `パス不一致 (encoded):\n  resolveURL → "${encodedMocPath}"\n  webkit     → "${encodedMocWebkit}"`
        )
      }

      // ── Override resolveURL to return encodeURI-wrapped paths ──
      // FileLoader.factory calls encodeURI(file.webkitRelativePath) when building
      // the lookup map, but compares against resolveURL() output directly.
      // Wrapping resolveURL ensures both sides are URI-encoded and match.
      // FileLoader overwrites resolveURL with its own blob resolver after
      // validateFiles() succeeds, so this override is only active during validation.
      const originalResolveURL = settings.resolveURL.bind(settings)
      settings.resolveURL = (path: string) => encodeURI(originalResolveURL(path))

      // ── PixiJS v7 Application ──
      const canvas = document.createElement('canvas')
      canvas.width  = LIVE2D_CANVAS_SIZE
      canvas.height = LIVE2D_CANVAS_SIZE

      const app = new PIXI.Application({
        view: canvas,
        width: LIVE2D_CANVAS_SIZE,
        height: LIVE2D_CANVAS_SIZE,
        backgroundAlpha: 0,
        preserveDrawingBuffer: true,
        antialias: true,
        powerPreference: 'low-power',
        autoDensity: false,
        resolution: 1,
        forceCanvas: false,
      } as any)

      if (!app.renderer || !app.view) {
        URL.revokeObjectURL(settingsObjectURL)
        throw new Error('PixiJS renderer の初期化に失敗しました')
      }

      pixiAppRef.current         = app
      pixiCanvasRef.current      = canvas
      settingsObjectURLRef.current = settingsObjectURL

      const fileArray = normalizedFiles as any
      fileArray.settings = settings
      const model = await (Live2DModel as any).from(fileArray, { autoInteract: false })
      app.stage.addChild(model)
      // Stage 4: drive actual Cubism model parameters, not the whole Pixi sprite.
      // The model's physics3.json uses the principal Angle/BodyAngle parameters as inputs.
      const internalModel = model.internalModel as any
      const coreModel = internalModel?.coreModel as any
      if (internalModel?.on && coreModel) {
        const rawParams = coreModel._model?.parameters
        const byId = new Map<string, { index: number; min: number; max: number }>()
        const count = typeof coreModel.getParameterCount === 'function'
          ? coreModel.getParameterCount() : (rawParams?.count ?? 0)
        for (let index = 0; index < count; index++) {
          const rawId = rawParams?.ids?.[index] ??
            (typeof coreModel.getParameterId === 'function' ? coreModel.getParameterId(index) : null)
          if (rawId == null) continue
          const id = typeof rawId?.getString === 'function' ? rawId.getString() : String(rawId)
          byId.set(id, {
            index,
            min: rawParams?.minimumValues?.[index] ?? -30,
            max: rawParams?.maximumValues?.[index] ?? 30,
          })
        }
        const tracked = [
          'ParamEyeLOpen','ParamEyeROpen','ParamMouthOpenY','ParamMouthForm',
          'ParamAngleX','ParamAngleY','ParamAngleZ',
          'ParamBodyAngleX','ParamBodyAngleY','ParamBodyAngleZ',
          'ParamEyeBallX','ParamEyeBallY', 'ParamEyesSmile',
          'ParamBrowLForm','ParamBrowRForm','ParamEyeBallForm',
        ]
        console.info('[FaceTrack] matched:', tracked.filter(id => byId.has(id)))
        const current = new Map<string, number>()
        const drive = (id: string, target: number, alpha = .28) => {
          const p = byId.get(id)
          if (!p) return
          const prev = current.get(id) ?? (rawParams?.values?.[p.index] ?? target)
          const value = Math.max(p.min, Math.min(p.max, prev + (target - prev) * alpha))
          current.set(id, value)
          if (typeof coreModel.setParameterValueByIndex === 'function') {
            coreModel.setParameterValueByIndex(p.index, value)
          } else if (rawParams?.values) rawParams.values[p.index] = value
        }
        internalModel.on('beforeModelUpdate', () => {
          const input = faceReadoutRef.current
          if (!input) return
          // Wide eyes may exceed 1 if the model supports it; honor the actual moc3 limits.
          drive('ParamEyeLOpen', Math.max(0, 1 - (input.leftBlink ?? input.blink) * 1.5 + (input.leftWide ?? 0) * .85), .55)
          drive('ParamEyeROpen', Math.max(0, 1 - (input.rightBlink ?? input.blink) * 1.5 + (input.rightWide ?? 0) * .85), .55)
          drive('ParamMouthOpenY', Math.min(1, input.mouth * 1.4), .48)
          // Mouth shape: smile versus pursed/funneled lips; no emotion classification.
          const smile = input.smile
          const pursed = Math.max(input.mouthPucker ?? 0, (input.mouthFunnel ?? 0) * .8)
          const stretched = input.mouthWide ?? 0
          drive('ParamMouthForm', Math.max(-1, Math.min(1, smile * 1.7 + stretched * .35 - pursed * 1.7 - .25)), .36)
          // Eye-smile is a dedicated modeled parameter. Keep blinks independent.
          const eyeSmile = Math.max(input.leftSmileEye ?? 0, input.rightSmileEye ?? 0)
          const smileEyesTarget = Math.max(0, Math.min(1, eyeSmile * .8 + Math.max(0, smile - .4) * .55))
          const eyeSmileParam = byId.get('ParamEyesSmile')
          if (eyeSmileParam) drive('ParamEyesSmile', smileEyesTarget * Math.max(1, eyeSmileParam.max), .32)
          // Brow form ranges differ by model, so map normalized facial movement into each actual range.
          const setBrow = (id: string, amount: number) => {
            const p = byId.get(id)
            if (!p) return
            const target = amount >= 0 ? amount * Math.max(0, p.max) : amount * Math.max(0, -p.min)
            drive(id, target, .30)
          }
          setBrow('ParamBrowLForm', (input.browLeft ?? 0) - (input.browFrown ?? 0) * .25)
          setBrow('ParamBrowRForm', (input.browRight ?? 0) - (input.browFrown ?? 0) * .25)
          // Head rotation — real model deformation, NOT whole-sprite motion.
          // Map normalized camera Euler angles onto the moc3 actual min/max.
          // Keep a responsive filter so head turns don't disappear under smoothing.
          const driveAngle = (id: string, normalized: number) => {
            const p = byId.get(id)
            if (!p) return
            const target = normalized >= 0 ? normalized * Math.max(0, p.max)
              : normalized * Math.max(0, -p.min)
            drive(id, target, .55)
          }
          driveAngle('ParamAngleX', input.headX ?? 0)
          driveAngle('ParamAngleY', input.headY ?? 0)
          driveAngle('ParamAngleZ', input.headZ ?? 0)
          drive('ParamEyeBallX', (input.eyeX ?? 0), .35)
          drive('ParamEyeBallY', (input.eyeY ?? 0), .35)
          // Crucial: separate body deformation from moving the entire avatar.
          drive('ParamBodyAngleX', (input.body?.sway ?? 0) * 10, .20)
          drive('ParamBodyAngleY', (input.body?.shrug ?? 0) * 7, .18)
          drive('ParamBodyAngleZ', (input.body?.tilt ?? 0) * 10, .20)
        })
      }

      await new Promise<void>(r => requestAnimationFrame(() => r()))

      const mw = model.width  || LIVE2D_CANVAS_SIZE
      const mh = model.height || LIVE2D_CANVAS_SIZE
      const sc = Math.min(
        (LIVE2D_CANVAS_SIZE * 0.9) / mw,
        (LIVE2D_CANVAS_SIZE * 0.9) / mh,
      )
      model.scale.set(sc)
      model.position.set(
        (LIVE2D_CANVAS_SIZE - mw * sc) / 2,
        (LIVE2D_CANVAS_SIZE - mh * sc) / 2,
      )

      loadedRef.current = true
      setStatus('loaded')

      await saveAvatarFiles(files)

    } catch (e) {
      const ne = parseNetworkError(e)
      let msg = ne.message
      if (ne.url)            msg += `\nURL: ${ne.url}`
      if (ne.status != null) msg += `\nHTTP status: ${ne.status}`
      if (ne.aborted)        msg += '\n(aborted)'

      setErrorMsg(msg)
      setDiagnostics(prev => prev ? {
        ...prev,
        networkErrorUrl:     ne.url,
        networkErrorStatus:  ne.status,
        networkErrorAborted: ne.aborted,
        networkErrorStack:   ne.stack,
      } : null)
      setStatus('error')

      if (pixiAppRef.current) {
        try { pixiAppRef.current.destroy(true) } catch { /* ignore */ }
        pixiAppRef.current = null
      }
      if (settingsObjectURLRef.current) {
        try { URL.revokeObjectURL(settingsObjectURLRef.current) } catch { /* ignore */ }
        settingsObjectURLRef.current = null
      }
      pixiCanvasRef.current = null
      loadedRef.current = false
    }
  }, [cleanup])

useEffect(() => {
  const restoreAvatar = async () => {
    try {
      const files = await loadAvatarFiles()

      if (!files || files.length === 0) return

      await loadModel(files)
    } catch (error) {
      console.error('前回のアバターの復元に失敗しました', error)
    }
  }

  restoreAvatar()
}, [loadModel])

useEffect(() => () => cleanup(), [cleanup])

const updateFace = useCallback((values: FaceReadout | null) => { faceReadoutRef.current = values }, [])
return { loadModel, cleanup, updateFace, pixiCanvasRef, loadedRef, status, errorMsg, modelName, diagnostics }
}