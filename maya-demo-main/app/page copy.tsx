"use client"

import { useState, useEffect, useRef } from "react"
import { useSpeechRecognition } from "react-speech-recognition"
import SpeechRecognition from "react-speech-recognition"
import { initializeSpeechRecognitionPolyfill } from "@/lib/speech-recognition-init"
import AnimatedSphere from "@/components/animated-sphere"
import AudioVisualizer from "@/components/audio-visualizer"
import VoiceControls from "@/components/voice-controls"
import { MessageCircle } from "lucide-react"

export default function Home() {
  const [polyfillReady, setPolyfillReady] = useState(false)
  const [fullTranscript, setFullTranscript] = useState("")
  const [response, setResponse] = useState("")
  const [isSending, setIsSending] = useState(false)
  const timerRef = useRef<number | null>(null)
  const transcriptRef = useRef<string>("")

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

  // Cleanup any pending timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        window.clearTimeout(timerRef.current)
        timerRef.current = null
      }
    }
  }, [])

  // Only call the hook after polyfill is initialized
  const { transcript, listening, resetTranscript, browserSupportsSpeechRecognition } = useSpeechRecognition()

  // keep a ref of the latest transcript so async handlers can read it
  useEffect(() => {
    transcriptRef.current = transcript || ""
  }, [transcript])

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
      // clear any pending timer
      if (timerRef.current) {
        window.clearTimeout(timerRef.current)
        timerRef.current = null
      }
      await handleAfterStop()
      return
    }
    console.log('Starting to listen for speech')
    try {
      // reset previous transcripts and start listening
      console.log('Resetting transcript and starting listening')
      resetTranscript()
      console.log('Transcript reset. Current transcript:', transcriptRef.current)
      await SpeechRecognition.startListening({ continuous: true, language: "en-US" })
      console.log('Started listening')
      // Stop listening automatically after 6 seconds
      timerRef.current = window.setTimeout(async () => {
        try {
          console.log('Auto-stopping listening after timeout')
          // await SpeechRecognition.stopListening()
          console.log('Stopped listening after timeout')
          await handleAfterStop()
          console.log('Handled after stop listening')
        } catch (e) {
          console.error('Error stopping/listening:', e)
        }
      }, 6000)
    } catch (e) {
      console.error('Failed to start listening:', e)
    }
  }

  // Called after stopping listening to process transcript and send to server
  const handleAfterStop = async () => {
    console.log('Handling after stop listening inside handleAfterStop')
    // Wait briefly for the hook to update final transcript after recognition stops
    const prev = transcriptRef.current
    const timeoutMs = 2000
    const intervalMs = 100
    let elapsed = 0
    while (elapsed < timeoutMs) {
      if (transcriptRef.current && transcriptRef.current !== prev) break
      // small delay
      // eslint-disable-next-line no-await-in-loop
      await new Promise((r) => setTimeout(r, intervalMs))
      elapsed += intervalMs
    }

    const text = (transcriptRef.current || "").trim()
    if (!text) return

    // append to local transcript
    setFullTranscript((prev) => (prev ? prev + " " + text : text))

    // send to backend
    setIsSending(true)
    try {
      console.log('Sending transcript to server:', text)
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
      setResponse(reply)
    } catch (err) {
      console.error('Failed to send transcript to server:', err)
      setResponse('Error: could not get response from server')
    } finally {
      setIsSending(false)
      // reset interim transcript so next listen starts fresh
      resetTranscript()
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

      <div className="relative z-10 flex flex-col items-center justify-center gap-8 max-w-2xl">
        {/* Animated Sphere */}
        <div className="relative">
          <AnimatedSphere isActive={listening} />
        </div>

        {/* Greeting Text */}
        <div className="text-center space-y-4">
          <p className="text-lg text-muted-foreground" style={{fontSize:40}}>
            <span className="text-muted-foreground/60">
              {((transcript.length)==0) && <span className="text-muted-foreground/60">Hi! I am Maya! Your Personal Assistant</span>}
              {fullTranscript && <span className="block mt-2 font-semibold text-foreground">{fullTranscript}</span>}
              {/* I am somehting that helps cardio vascular health. I am learning to become an expert to post canser  */}
              {transcript && !fullTranscript && <span className="block mt-2 font-semibold text-foreground">{transcript}</span>}
            </span>{" "}
            {/* <span className="font-semibold text-foreground">How can I help you?</span> */}
          </p>

          {/* Display response */}
          {response && (
            <div className="flex items-start gap-2 mt-4 p-4 bg-emerald-50 rounded-lg border border-emerald-200">
              <MessageCircle className="w-5 h-5 text-emerald-600 mt-0.5 flex-shrink-0" />
              <p className="text-sm text-emerald-800">{response}</p>
            </div>
          )}
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
