export const initializeSpeechRecognitionPolyfill = async () => {
  // Only run in browser environment
  if (typeof window === 'undefined') {
    console.log('Not in browser environment, skipping polyfill initialization')
    return
  }

  const region = process.env.NEXT_PUBLIC_AZURE_SPEECH_REGION
  const subscriptionKey = process.env.NEXT_PUBLIC_AZURE_SPEECH_KEY

  console.log('Initializing speech recognition polyfill with region:', region)

  // Check if polyfill is already applied or if we have valid credentials
  if (!region || !subscriptionKey || subscriptionKey === 'your_subscription_key_here') {
    console.warn(
      'Azure Speech Services polyfill not configured. Please set NEXT_PUBLIC_AZURE_SPEECH_REGION and NEXT_PUBLIC_AZURE_SPEECH_KEY in .env.local',
      { region, hasKey: !!subscriptionKey }
    )
    return
  }

  try {
    console.log('Importing web-speech-cognitive-services...')
    // Dynamic imports to avoid server-side execution
    const { createSpeechServicesPonyfill } = await import('web-speech-cognitive-services')
    const SpeechRecognition = (await import('react-speech-recognition')).default

    console.log('Creating Azure speech recognition ponyfill...')
    const { SpeechRecognition: AzureSpeechRecognition } = await createSpeechServicesPonyfill({
      credentials: {
        region,
        subscriptionKey,
      },
    })

    console.log('Applying polyfill to react-speech-recognition...')
    SpeechRecognition.applyPolyfill(AzureSpeechRecognition)
    console.log('✅ Azure Speech Recognition polyfill applied successfully')
  } catch (error) {
    console.error('❌ Failed to apply Azure Speech Recognition polyfill:', error)
    throw error
  }
}
