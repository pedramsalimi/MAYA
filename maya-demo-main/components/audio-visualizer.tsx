"use client"

interface AudioVisualizerProps {
  isActive: boolean
}

export default function AudioVisualizer({ isActive }: AudioVisualizerProps) {
  const bars = Array.from({ length: 8 })

  return (
    <div className="flex items-center justify-center gap-2 py-8">
      {bars.map((_, i) => (
        <div
          key={i}
          className="w-1.5 bg-gradient-to-t from-primary to-accent rounded-full"
          style={{
            height: "24px",
            animation: isActive ? `wave 0.8s ease-in-out ${i * 0.1}s infinite` : "none",
            opacity: isActive ? 1 : 0.3,
          }}
        >
          <style>{`
            @keyframes wave {
              0%, 100% { height: 24px; }
              50% { height: 48px; }
            }
          `}</style>
        </div>
      ))}
    </div>
  )
}
