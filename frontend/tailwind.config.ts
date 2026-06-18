import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#172033",
        panel: "#f7f8fb",
        line: "#d9dee8"
      },
      boxShadow: {
        focus: "0 0 0 3px rgba(44, 123, 229, 0.18)"
      }
    }
  },
  plugins: []
};

export default config;
