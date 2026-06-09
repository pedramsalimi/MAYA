// Azure Text-to-Speech utility for browser
// Usage: await speakWithAzureTTS(text)

let speechSynthesizer: any = null

export async function speakWithAzureTTS(text: string) {
  if (!text) return
  const region = process.env.NEXT_PUBLIC_AZURE_SPEECH_REGION
  const key = process.env.NEXT_PUBLIC_AZURE_SPEECH_KEY
  if (!region || !key) {
    console.warn('Azure TTS: Missing region or key')
    return
  }
  if (typeof window === 'undefined') return
  if (!speechSynthesizer) {
    const sdk = await import('microsoft-cognitiveservices-speech-sdk')
    const speechConfig = sdk.SpeechConfig.fromSubscription(key, region)
    speechConfig.speechSynthesisVoiceName = 'en-US-JennyNeural' // You can change the voice here
    // Use MP3 output for best compatibility with Chrome and other browsers
    speechConfig.speechSynthesisOutputFormat = sdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    const audioConfig = sdk.AudioConfig.fromDefaultSpeakerOutput()
    speechSynthesizer = new sdk.SpeechSynthesizer(speechConfig, audioConfig)
  }
  return new Promise<void>((resolve, reject) => {
    speechSynthesizer.speakTextAsync(
      text,
      () => {
        console.log('Azure TTS: Finished speaking')
        resolve()
      },
      (err: any) => {
        console.error('Azure TTS error', err)
        reject(err)
      }
    )
  })
}