declare module 'react-speech-recognition' {
  interface UseSpeechRecognitionOptions {
    transcribing?: boolean
    clearTranscriptOnListen?: boolean
    commands?: any[]
  }

  interface UseSpeechRecognitionReturn {
    transcript: string
    interimTranscript: string
    finalTranscript: string
    isMicrophoneAvailable: boolean
    browserSupportsSpeechRecognition: boolean
    browserSupportsContinuousListening: boolean
    listening: boolean
    resetTranscript: () => void
    abortListening: () => Promise<void>
    startListening: (options?: { continuous?: boolean; language?: string }) => Promise<void>
    stopListening: () => Promise<void>
  }

  export function useSpeechRecognition(options?: UseSpeechRecognitionOptions): UseSpeechRecognitionReturn

  const SpeechRecognition: {
    counter: number
    applyPolyfill: (polyfill: any) => void
    removePolyfill: () => void
    getRecognitionManager: () => any
    getRecognition: () => any
    startListening: (options?: { continuous?: boolean; language?: string }) => Promise<void>
    stopListening: () => Promise<void>
    abortListening: () => Promise<void>
    browserSupportsSpeechRecognition: () => boolean
    browserSupportsContinuousListening: () => boolean
  }

  export default SpeechRecognition
}
