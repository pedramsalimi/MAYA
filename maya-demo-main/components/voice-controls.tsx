"use client"

import { Mic, MicOff } from "lucide-react"
import { Button } from "@/components/ui/button"

interface VoiceControlsProps {
  isListening: boolean
  onToggle: () => void
  isBusy?: boolean
}

export default function VoiceControls({ isListening, onToggle, isBusy }: VoiceControlsProps) {
  return (
    <div className="flex flex-col items-center gap-4">
      <Button
        onClick={onToggle}
        disabled={isBusy}
        size="lg"
        className={`rounded-full w-16 h-16 transition-all duration-300 ${
          isListening
            ? "bg-primary hover:bg-primary/90 text-primary-foreground"
            : "bg-secondary hover:bg-secondary/90 text-secondary-foreground"
        }`}
      >
        {isListening ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
      </Button>
      <p className="text-2xl font-semibold text-muted-foreground">{isBusy ? "Thinking..." : isListening ? "Listening..." : "Click to start talking"}</p>
    </div>
  )
}
