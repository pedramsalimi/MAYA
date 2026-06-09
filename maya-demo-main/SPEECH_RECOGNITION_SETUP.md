# Setting Up Azure Speech Services Polyfill

This guide explains how to set up the Azure Speech Services polyfill for cross-browser speech recognition support.

## Why Use a Polyfill?

- ✅ Voice-enabled on all modern browsers (except Internet Explorer)
- ✅ Consistent voice experience across browsers
- ✅ You control who processes your users' voice data
- ✅ Suitable for commercial applications

## Step-by-Step Setup

### 1. Create an Azure Account

1. Go to [Azure Portal](https://portal.azure.com/)
2. Sign up for a free account (includes $200 free credits)

### 2. Create a Speech Services Resource

1. In the Azure Portal, click "Create a resource"
2. Search for "Speech"
3. Click "Speech" → "Create"
4. Fill in the form:
   - **Resource Group**: Create new or select existing
   - **Region**: Choose a region closest to your users (e.g., `eastus`, `westeurope`, `southeastasia`)
   - **Name**: Give it a meaningful name
   - **Pricing tier**: Select "Free F0" (limited) or "Standard S0" (recommended)
5. Click "Review + create" → "Create"

### 3. Get Your Credentials

1. Once created, go to the Speech resource
2. Under "Keys and Endpoint", copy:
   - **Region**: (e.g., `eastus`)
   - **Key 1** or **Key 2**: (your subscription key)

### 4. Configure Your App

1. Open `.env.local` in your project root
2. Replace the placeholder values:

```env
NEXT_PUBLIC_AZURE_SPEECH_REGION=eastus
NEXT_PUBLIC_AZURE_SPEECH_KEY=your_actual_subscription_key_here
```

3. Restart your dev server

## Supported Regions

Common regions:
- `eastus` - East US
- `westus` - West US
- `westeurope` - West Europe
- `northeurope` - North Europe
- `southeastasia` - Southeast Asia
- `eastasia` - East Asia

See [all supported regions](https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/regions)

## Troubleshooting

### Still seeing "Browser doesn't support speech recognition"

1. Check that your `.env.local` file has valid credentials
2. Restart the dev server
3. Clear browser cache and reload the page
4. Check browser console for error messages

### "Not-allowed" error

- The browser is denying microphone access
- Grant microphone permissions when prompted
- Check browser settings to ensure microphone access is allowed

### Azure API errors

- Verify your subscription key is correct (no extra spaces)
- Check that your region matches the resource's region
- Ensure your Azure subscription is active and has billing set up

## Privacy & Security

⚠️ **Important**: 

- `NEXT_PUBLIC_*` variables are exposed to the browser
- This is necessary for client-side speech recognition
- The Azure Cognitive Services will process voice data according to [Microsoft's privacy policy](https://privacy.microsoft.com/)
- For production, consider using a backend proxy to avoid exposing keys

## Cost Estimation

- **Free tier (F0)**: 5,000 API calls/month
- **Standard tier (S0)**: $7.50/month for pay-as-you-go
- **Expected usage**: ~1-2 API calls per minute of speech

## Additional Resources

- [Azure Speech Services Documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/)
- [react-speech-recognition GitHub](https://github.com/JamesBrill/react-speech-recognition)
- [Web Speech Cognitive Services Polyfill](https://github.com/azure-sdk/cognitive-services-speech-sdk-js)
