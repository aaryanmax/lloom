@echo off
REM llama.cpp server for Qwen3-8B Q4_K_M — partial GPU offload for a 4GB card.
REM 8B Q4_K_M is ~4.87GB; -ngl 25 keeps most layers in VRAM, rest in system RAM.
REM KV cache quantized to q8_0 to save additional VRAM.
"C:\Users\Admin\Projects\LocalAi\llama.cpp\llama-server.exe" ^
  -m "C:\Users\Admin\Projects\LocalAi\models\qwen3-8b-q4_K_M.gguf" ^
  -ngl %QWEN_NGL% ^
  -c 8192 ^
  -ctk q8_0 -ctv q8_0 ^
  --threads 8 ^
  --host 127.0.0.1 --port %QWEN_PORT%
