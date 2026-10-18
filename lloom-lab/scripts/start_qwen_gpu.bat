@echo off
set "PATH=C:\Users\Admin\Projects\.supervenv\Lib\site-packages\torch\lib;%PATH%"
"C:\Users\Admin\Projects\LocalAi\llama.cpp\llama-server.exe" -m "C:\Users\Admin\Projects\LocalAi\models\qwen3-8b-q4_K_M.gguf" -a qwen3 -ngl 16 -c 8192 --parallel 2 -ctk q8_0 -ctv q8_0 --threads 12 --host 127.0.0.1 --port 8123
