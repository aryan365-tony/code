/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dark-bg': '#0D1117', // Deep Space/GitHub dark background
        'accent-glow': '#00ffff', // Electric Cyan Neon
        'text-primary': '#e6edf3', // Light text for contrast
        'secondary-text': '#8b949e', // Subdued text
      },
      boxShadow: {
        'glow': '0 0 10px rgba(0, 255, 255, 0.3)',
      },
      animation: {
        'pulse-glow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}