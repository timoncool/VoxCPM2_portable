@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   VoxCPM2 - Мультиязычный TTS
echo   Портативная русскоязычная версия
echo ========================================
echo.

REM Определяем директорию скрипта
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Проверяем наличие Python
if not exist "python\python.exe" (
    echo ОШИБКА: Python не найден!
    echo.
    echo Пожалуйста, сначала запустите install.bat для установки.
    echo.
    pause
    exit /b 1
)

REM Проверяем наличие app.py
if not exist "app.py" (
    echo ОШИБКА: Файл app.py не найден!
    echo.
    pause
    exit /b 1
)

REM Читаем версию CUDA из конфигурационного файла
if exist "cuda_version.txt" (
    set /p CUDA_VERSION=<cuda_version.txt
    echo Конфигурация: !CUDA_VERSION!
    echo.
)

REM =====================================================
REM ИЗОЛЯЦИЯ: Все временные и кэш файлы в папке приложения
REM =====================================================

REM Локальные temp директории
set "TEMP=%SCRIPT_DIR%temp"
set "TMP=%SCRIPT_DIR%temp"
set "GRADIO_TEMP_DIR=%SCRIPT_DIR%temp"
if not exist "%TEMP%" mkdir "%TEMP%"

REM Hugging Face кэш и модели в локальной папке
set "HF_HOME=%SCRIPT_DIR%models"
set "HUGGINGFACE_HUB_CACHE=%SCRIPT_DIR%models"
set "TRANSFORMERS_CACHE=%SCRIPT_DIR%models"
set "HF_DATASETS_CACHE=%SCRIPT_DIR%models\datasets"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"

REM Torch кэш
set "TORCH_HOME=%SCRIPT_DIR%models\torch"
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%"

REM ModelScope кэш (используется voxcpm для ZipEnhancer/FunASR)
set "MODELSCOPE_CACHE=%SCRIPT_DIR%models\modelscope"
if not exist "%MODELSCOPE_CACHE%" mkdir "%MODELSCOPE_CACHE%"

REM XDG кэш (используется некоторыми библиотеками)
set "XDG_CACHE_HOME=%SCRIPT_DIR%cache"
if not exist "%XDG_CACHE_HOME%" mkdir "%XDG_CACHE_HOME%"

REM FFmpeg в PATH (для обработки аудио)
if exist "%SCRIPT_DIR%ffmpeg\ffmpeg.exe" (
    set "PATH=%SCRIPT_DIR%ffmpeg;%PATH%"
    echo FFmpeg: Локальный портативный
) else (
    echo ВНИМАНИЕ: FFmpeg не найден. Некоторые форматы могут не работать.
)

REM Папка для выходных файлов
set "OUTPUT_DIR=%SCRIPT_DIR%output"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM =====================================================
REM Переменные окружения для Python
REM =====================================================
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set "PYTHONPATH=%SCRIPT_DIR%"
set TOKENIZERS_PARALLELISM=false

REM Запускаем приложение
echo Запуск VoxCPM2...
echo.
echo При первом запуске модель VoxCPM2 (~4-5 GB) скачается автоматически
echo в папку models\. Подождите пока загрузится.
echo.
echo После запуска приложение будет доступно по адресу:
echo.
echo    http://127.0.0.1:7860
echo.
echo Для остановки приложения нажмите Ctrl+C
echo.
echo ========================================
echo.

python\python.exe app.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo ОШИБКА при запуске приложения!
    echo ========================================
    echo.
    echo Возможные причины:
    echo 1. Не установлены зависимости - запустите install.bat
    echo 2. Недостаточно памяти GPU (нужно минимум 8GB VRAM для VoxCPM2)
    echo 3. Проблемы с CUDA драйверами
    echo 4. Модели не загружены (требуется интернет при первом запуске)
    echo.
    pause
    exit /b 1
)

pause
