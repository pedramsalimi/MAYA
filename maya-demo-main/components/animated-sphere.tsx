"use client"

import { useEffect, useRef } from "react"

interface AnimatedSphereProps {
  isActive?: boolean
}

export default function AnimatedSphere({ isActive = false }: AnimatedSphereProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    canvas.width = 300
    canvas.height = 300

    const centerX = canvas.width / 2
    const centerY = canvas.height / 2
    const radius = 100
    let rotation = 0

    const drawSphere = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Create gradient
      const gradient = ctx.createRadialGradient(centerX - 30, centerY - 30, 0, centerX, centerY, radius)

      gradient.addColorStop(0, "rgba(220, 200, 255, 0.8)")
      gradient.addColorStop(0.5, "rgba(150, 100, 255, 0.6)")
      gradient.addColorStop(1, "rgba(100, 50, 200, 0.7)")

      // Draw sphere
      ctx.fillStyle = gradient
      ctx.beginPath()
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2)
      ctx.fill()

      // Add shine effect
      const shineGradient = ctx.createRadialGradient(centerX - 40, centerY - 40, 0, centerX, centerY, radius * 1.5)

      shineGradient.addColorStop(0, "rgba(255, 255, 255, 0.4)")
      shineGradient.addColorStop(0.5, "rgba(255, 255, 255, 0.1)")
      shineGradient.addColorStop(1, "rgba(255, 255, 255, 0)")

      ctx.fillStyle = shineGradient
      ctx.beginPath()
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2)
      ctx.fill()

      // Add animated shimmer when listening
      if (isActive) {
        ctx.strokeStyle = `rgba(200, 150, 255, ${0.3 + Math.sin(rotation / 10) * 0.2})`
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(centerX, centerY, radius + 10, 0, Math.PI * 2)
        ctx.stroke()
      }

      // Update rotation
      rotation += isActive ? 2 : 0.5
      animationRef.current = requestAnimationFrame(drawSphere)
    }

    drawSphere()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isActive])

  return (
    <div className="relative w-80 h-80 flex items-center justify-center">
      <div
        className={`absolute inset-0 rounded-full transition-all duration-300 ${isActive ? "animate-pulse-ring" : ""}`}
      />
      <canvas ref={canvasRef} className="drop-shadow-2xl animate-float" />
    </div>
  )
}
