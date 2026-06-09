"use client"
import { useState, useEffect, useRef } from "react"
import { useSpeechRecognition } from "react-speech-recognition"
import SpeechRecognition from "react-speech-recognition"
import { initializeSpeechRecognitionPolyfill } from "@/lib/speech-recognition-init"
import { speakWithAzureTTS } from "@/lib/azure-tts"
import AnimatedSphere from "@/components/animated-sphere"
import AudioVisualizer from "@/components/audio-visualizer"
import VoiceControls from "@/components/voice-controls"
import { MessageCircle } from "lucide-react"
export default function Home() {
const [polyfillReady, setPolyfillReady] = useState(false)
const [fullTranscript, setFullTranscript] = useState("")
const [response, setResponse] = useState("")
const [isSending, setIsSending] = useState(false)
const [isSpeaking, setIsSpeaking] = useState(false)
const silenceTimerRef = useRef<number | null>(null)
const transcriptRef = useRef<string>("")
const ignoreTranscriptRef = useRef(false)
const ignoreTimerRef = useRef<number | null>(null)
const ttsCleanupIntervalRef = useRef<number | null>(null)
const ttsSpokenRef = useRef<string>("")
const processedTranscriptsRef = useRef<Set<string>>(new Set())
const dropBacklogTranscriptsRef = useRef(false)
const isListeningRef = useRef(false)
const isSpeakingRef = useRef(false)
const isSendingRef = useRef(false)
const isProcessingRef = useRef(false)
const [hasSpokenBefore, setHasSpokenBefore] = useState(false)
const [currentUserMessage, setCurrentUserMessage] = useState("")
const [currentResponse, setCurrentResponse] = useState("")
// Initialize polyfill first, then set ready state
useEffect(() => {
 const init = async () => {
   await initializeSpeechRecognitionPolyfill()
   // Give a small delay to ensure polyfill is fully applied
   await new Promise((resolve) => setTimeout(resolve, 500))
   setPolyfillReady(true)
 }
 init()
}, [])
// Cleanup any pending silence timer on unmount
useEffect(() => {
 return () => {
   if (silenceTimerRef.current) {
     window.clearTimeout(silenceTimerRef.current)
     silenceTimerRef.current = null
   }
   if (ignoreTimerRef.current) {
     window.clearTimeout(ignoreTimerRef.current)
     ignoreTimerRef.current = null
   }
   if (ttsCleanupIntervalRef.current) {
     window.clearInterval(ttsCleanupIntervalRef.current)
     ttsCleanupIntervalRef.current = null
   }
   dropBacklogTranscriptsRef.current = false
 }
}, [])
// Only call the hook after polyfill is initialized
const { transcript, listening, resetTranscript, browserSupportsSpeechRecognition } = useSpeechRecognition()
// keep a ref of the latest transcript so async handlers can read it
useEffect(() => {
 // Ignore transcript updates while the app is speaking (TTS) or while we're
 // sending the transcript to the server. This prevents capturing user
 // speech that happens during TTS playback or during server processing.
 // Also ignore if we're in an explicit 'ignore' window (set when TTS starts)
 if (isSpeakingRef.current || isSendingRef.current || ignoreTranscriptRef.current) {
   console.log('Ignoring transcript update while speaking/sending/ignoring')
   return
 }








 const ttsText = ttsSpokenRef.current || ""
 const candidate = (transcript || "").trim()








 if (dropBacklogTranscriptsRef.current) {
   if (candidate) {
     console.log('Dropping transcript captured during TTS:', candidate)
   } else {
     console.log('Transcript buffer cleared during TTS ignore window')
   }
   transcriptRef.current = ""
   resetTranscript()
   return
 }








 if (ttsText) {
   try {
     const tLower = ttsText.toLowerCase()
     const cLower = candidate.toLowerCase()
     if (cLower && (cLower === tLower || tLower.includes(cLower) || cLower.includes(tLower))) {
       console.log('Ignoring transcript because it matches last TTS output')
       return
     }
   } catch (e) {
     // If any unexpected error occurs in matching, fall back to normal behavior
     console.warn('Error comparing transcript to TTS text', e)
   }
 }








 // Reject transcripts we've already processed in this session
 const candidateLower = candidate.toLowerCase()
 if (candidateLower && processedTranscriptsRef.current.has(candidateLower)) {
   console.log('Ignoring transcript because it was already processed:', candidate)
   return
 }








 transcriptRef.current = transcript || ""
 console.log('transcriptRef updated:', transcriptRef.current)
}, [transcript])








// Track listening state in ref
useEffect(() => {
 isListeningRef.current = listening
 console.log('🎤 Listening state changed:', listening)
}, [listening])








// Track isSpeaking in ref for silence detection
useEffect(() => {
 isSpeakingRef.current = isSpeaking
}, [isSpeaking])








// Track isSending in ref for silence detection
useEffect(() => {
 isSendingRef.current = isSending
}, [isSending])








// Manage silence detection lifecycle
useEffect(() => {
 if (!listening) {
   // Stop silence detection when not listening
   if (silenceTimerRef.current) {
     window.clearTimeout(silenceTimerRef.current)
     silenceTimerRef.current = null
   }
   // Call handleAfterStop only if we haven't already processed this session
   if (!isProcessingRef.current && transcriptRef.current.trim()) {
     console.log('Listening stopped and not processing, calling handleAfterStop')
     isProcessingRef.current = true
     handleAfterStop()
   }
   return
 }








 // Start silence detection when listening
 console.log('useEffect: Listening started - starting silence detection')
 // Only reset processing flag if this is a manual start (new mic click)
 // Don't reset if it's just clearing transcript after TTS
 startSilenceDetection()








 return () => {
   if (silenceTimerRef.current) {
     window.clearTimeout(silenceTimerRef.current)
     silenceTimerRef.current = null
   }
 }
}, [listening])
// Show loading state while polyfill is initializing
if (!polyfillReady) {
 return (
   <main className="min-h-screen bg-gradient-to-br from-background via-secondary/5 to-accent/10 flex items-center justify-center p-4">
     <div className="relative z-10 flex flex-col items-center justify-center gap-4">
       <AnimatedSphere isActive={true} />
       <p className="text-muted-foreground animate-pulse">Initializing speech recognition...</p>
     </div>
   </main>
 )
}
if (!browserSupportsSpeechRecognition) {
 return (
   <main className="min-h-screen bg-gradient-to-br from-background via-secondary/5 to-accent/10 flex items-center justify-center p-4">
     <div className="relative z-10 flex flex-col items-center justify-center gap-8 max-w-2xl">
       <div className="text-center space-y-4">
         <h1 className="text-2xl font-bold text-foreground">Browser Speech Recognition Not Supported</h1>
         <p className="text-lg text-muted-foreground">
           Your browser doesn't support Web Speech API, and the polyfill couldn't be initialized.
         </p>
         <div className="mt-6 p-4 bg-red-50 dark:bg-red-950 rounded-lg border border-red-200 dark:border-red-800">
           <p className="text-sm text-red-800 dark:text-red-200">
             <strong>Please try:</strong>
           </p>
           <ol className="text-sm text-red-800 dark:text-red-200 mt-2 space-y-2 list-decimal list-inside">
             <li>Open the browser console (F12) to check for errors</li>
             <li>Verify your Azure credentials are correct in <code className="bg-red-100 dark:bg-red-900 px-2 py-1 rounded">.env.local</code></li>
             <li>Restart the development server</li>
             <li>Try a different browser (Chrome, Edge, or Firefox work best)</li>
             <li>Check that your Azure Speech Services region is correct</li>
           </ol>
         </div>
         <details className="mt-6 text-left">
           <summary className="cursor-pointer text-sm font-semibold text-muted-foreground hover:text-foreground">
             Debug Information
           </summary>
           <div className="mt-2 p-3 bg-slate-100 dark:bg-slate-900 rounded text-xs font-mono text-slate-900 dark:text-slate-100 overflow-auto max-h-60">
             <p>Browser Support: {String(browserSupportsSpeechRecognition)}</p>
             <p>Polyfill Ready: {String(polyfillReady)}</p>
             <p>User Agent: {typeof navigator !== 'undefined' ? navigator.userAgent : 'N/A'}</p>
           </div>
         </details>
       </div>
     </div>
   </main>
 )
}
const handleToggle = async () => {
 console.log('Toggling listening state. Currently listening:', listening)
 if (listening) {
   // If already listening, stop early
   await SpeechRecognition.stopListening()
   return
 }
 console.log('Starting to listen for speech')
 try {
   clearTranscriptBuffer()
   await SpeechRecognition.startListening({ continuous: true, language: "en-US" })
 } catch (e) {
   console.error('Failed to start listening:', e)
 }
}








const clearTranscriptBuffer = (opts?: { clearProcessed?: boolean }) => {
 resetTranscript()
 transcriptRef.current = ""
 if (opts?.clearProcessed !== false) {
   processedTranscriptsRef.current.clear()
 }
}








const stopTtsCleanupInterval = () => {
 if (ttsCleanupIntervalRef.current) {
   window.clearInterval(ttsCleanupIntervalRef.current)
   ttsCleanupIntervalRef.current = null
 }
}








const startTtsCleanupInterval = () => {
 stopTtsCleanupInterval()
 ttsCleanupIntervalRef.current = window.setInterval(() => {
   // Keep wiping the transcript buffer while TTS audio plays so any
   // mic feedback never accumulates.
   clearTranscriptBuffer({ clearProcessed: false })
 }, 400)
}








// Start silence detection (2s silence + 2s confirmation = 4s total)
const startSilenceDetection = () => {
 console.log('startSilenceDetection called - waiting for user to speak and then go silent')
 console.log('Initial transcriptRef.current:', transcriptRef.current)
 let lastTranscript = ""
 let silenceDuration = 0
 const checkInterval = 300
 const silenceThreshold = 2000
 const confirmThreshold = 2000
  const silenceCheck = () => {
   // If we're currently speaking (TTS), sending to server, or in an ignore
   // window, don't count silence — reset counters and wait.
   if (isSpeakingRef.current || isSendingRef.current || ignoreTranscriptRef.current) {
     lastTranscript = transcriptRef.current || ""
     silenceDuration = 0
     silenceTimerRef.current = window.setTimeout(silenceCheck, checkInterval)
     return
   }








   const currentTranscript = transcriptRef.current || ""
    if (silenceDuration === 0) {
     console.log('First silenceCheck: currentTranscript="' + currentTranscript + '", lastTranscript="' + lastTranscript + '"')
   }
    // If transcript changed, user is speaking - reset silence counter
   if (currentTranscript !== lastTranscript) {
     console.log('📝 User speaking! Transcript changed from "' + lastTranscript + '" to "' + currentTranscript + '"')
     lastTranscript = currentTranscript
     silenceDuration = 0
     silenceTimerRef.current = window.setTimeout(silenceCheck, checkInterval)
     return
   }
    // Transcript hasn't changed, accumulate silence duration
   silenceDuration += checkInterval
   console.log('⏱️ Silence duration:', silenceDuration, '| transcript:"' + currentTranscript + '"')
    if (silenceDuration >= silenceThreshold + confirmThreshold) {
     // 4 seconds of silence total
     // Don't stop listening, just process the transcript
     if (currentTranscript.trim()) {
       console.log('✅ 4 seconds of silence detected, processing transcript')
       if (silenceTimerRef.current) {
         window.clearTimeout(silenceTimerRef.current)
         silenceTimerRef.current = null
       }
       // Process but don't stop listening
       handleAfterStop()
       // Reset for next cycle
       isProcessingRef.current = false
       lastTranscript = ""
       silenceDuration = 0
     }
     silenceTimerRef.current = window.setTimeout(silenceCheck, checkInterval)
     return
   }
    silenceTimerRef.current = window.setTimeout(silenceCheck, checkInterval)
 }
  silenceTimerRef.current = window.setTimeout(silenceCheck, checkInterval)
}
// Called after the user stops speaking (detected via silence) to process
// transcript and send to server
const handleAfterStop = async () => {
 console.log('handleAfterStop: Processing transcript after silence')
  // Wait for final transcript update
 const prev = transcriptRef.current
 const timeoutMs = 1000
 const intervalMs = 50
 let elapsed = 0
 while (elapsed < timeoutMs) {
   if (transcriptRef.current && transcriptRef.current !== prev) break
   await new Promise((r) => setTimeout(r, intervalMs))
   elapsed += intervalMs
 }








 const text = (transcriptRef.current || "").trim()
 console.log('Final transcript:', text)
 if (!text) {
   console.log('No transcript, returning')
   return
 }








 // Remember this transcript so we don't process it again if it resurfaces
 processedTranscriptsRef.current.add(text.toLowerCase())








 // Mark that user has spoken
 setHasSpokenBefore(true)
 setCurrentUserMessage(text)
 setCurrentResponse("")








 // Send to backend
 console.log('Sending to server:', text)
 setIsSending(true)
 try {
   const apiBase = 'http://127.0.0.1:8000'
   const res = await fetch(`${apiBase.replace(/\/+$/, '')}/conversation`, {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({ user_message: text }),
   })
   if (!res.ok) {
     const txt = await res.text()
     throw new Error(`Server error ${res.status}: ${txt}`)
   }
   const data = await res.json()
   const reply = data.response || data.message || ''
   setCurrentResponse(reply)
   console.log('Server reply received:', reply)
   setIsSending(false)
    // Speak the response using Azure TTS
   if (reply) {
     try {
       console.log('Starting TTS while keeping mic active...')
    
       // Keep listening continuously, but ignore and clear transcripts so we
       // don't capture TTS playback or user speech during assistant response.
       ignoreTranscriptRef.current = true
       dropBacklogTranscriptsRef.current = true
       ttsSpokenRef.current = reply
       if (ignoreTimerRef.current) {
         window.clearTimeout(ignoreTimerRef.current)
         ignoreTimerRef.current = null
       }
       clearTranscriptBuffer()
       startTtsCleanupInterval()








       isSpeakingRef.current = true
       setIsSpeaking(true)
       await speakWithAzureTTS(reply)
       console.log('TTS finished!')
       isSpeakingRef.current = false
       setIsSpeaking(false)
       stopTtsCleanupInterval()








       // Small delay to ensure TTS is fully done
       await new Promise((r) => setTimeout(r, 500))








       // Clear out any buffered transcript generated during TTS playback
       clearTranscriptBuffer()








       // Keep ignoring for a grace period after TTS to discard buffered
       // results made while we were speaking
       const graceMs = 1500
       ignoreTimerRef.current = window.setTimeout(() => {
         clearTranscriptBuffer()
         ignoreTranscriptRef.current = false
         dropBacklogTranscriptsRef.current = false
         ttsSpokenRef.current = ""
         if (ignoreTimerRef.current) {
           window.clearTimeout(ignoreTimerRef.current)
           ignoreTimerRef.current = null
         }
         console.log('✅ Ignore window cleared; ready for new input')
       }, graceMs)








       console.log('Ready for next user input')
     } catch (ttsErr) {
       console.error('TTS error:', ttsErr)
       stopTtsCleanupInterval()
       setIsSpeaking(false)
       isSpeakingRef.current = false
       // Make sure we don't stay in ignore mode if TTS failed
       clearTranscriptBuffer()
       ignoreTranscriptRef.current = false
       dropBacklogTranscriptsRef.current = false
       ttsSpokenRef.current = ""
       if (ignoreTimerRef.current) {
         window.clearTimeout(ignoreTimerRef.current)
         ignoreTimerRef.current = null
       }
     }
   }
 } catch (err) {
   console.error('Server error:', err)
   setCurrentResponse('Error: could not get response')
   setIsSending(false)
 }
}








return (
 <main className="min-h-screen bg-gradient-to-br from-background via-secondary/5 to-accent/10 flex items-center justify-center p-4">
   {/* Dot pattern background */}
   <div className="absolute inset-0 overflow-hidden pointer-events-none">
     <div className="absolute inset-0 opacity-30">
       {Array.from({ length: 100 }).map((_, i) => (
         <div
           key={i}
           className="absolute w-1 h-1 bg-primary/20 rounded-full"
           style={{
             left: `${Math.random() * 100}%`,
             top: `${Math.random() * 100}%`,
           }}
         />
       ))}
     </div>
   </div>








   <div className="relative z-10 flex flex-col items-center justify-center gap-8 max-w-6xl">
     {/* Animated Sphere */}
     <div className="relative">
       <AnimatedSphere isActive={listening} />
     </div>
     {/* Greeting Text */}
     <div className="text-center space-y-4">
       <div className="text-muted-foreground/90 space-y-4">
         {/* Show greeting only if user hasn't spoken yet */}
         {!hasSpokenBefore && (
           <div className="max-w-3xl mx-auto space-y-4 text-left">
             <p className="text-3xl font-semibold text-foreground">
               Hello! I’m Maya — your smart heart-health companion.
             </p>
             <p className="text-lg leading-relaxed">
               I’m part of a new Living Lab helping young adults stay on top of their heart health.
               Ask me about wellbeing, healthy habits, monitoring, or how technology can support personalised care.
             </p>
             <div>
               <p className="text-base font-semibold uppercase tracking-wide text-foreground/80">Try:</p>
               <ul className="mt-2 space-y-2 text-base text-foreground/80 list-disc list-inside">
                 <li>“How can I look after my heart day to day?”</li>
                 <li>“Why does heart monitoring matter?”</li>
                 <li>“What symptoms should I watch for?”</li>
                  <li>"Can you get my biomarkers?”</li>
               </ul>
             </div>
           </div>
         )}
         {/* Show current user message and response only (not history) */}
         {hasSpokenBefore && currentUserMessage && <span className="block mt-2 text-4xl font-semibold text-foreground">{currentUserMessage}</span>}
         {/* Show interim transcript while speaking */}
         {transcript && !currentUserMessage && !isSpeaking && <span className="block mt-2 text-4xl font-semibold text-foreground">{transcript}</span>}
         {/* Display current response in gray (same style as user text) */}
         {currentResponse && <span className="block mt-2 text-4xl font-semibold text-gray-500">{currentResponse}</span>}
       </div>
     </div>
     {/* Audio Visualizer */}
     {listening && (
       <div className="animate-in fade-in duration-300">
         <AudioVisualizer isActive={listening} />
       </div>
     )}
     {/* Voice Controls */}
     <VoiceControls isListening={listening} onToggle={handleToggle} isBusy={isSending} />
   </div>
 </main>
)
}



