@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   VoxCPM2 Portable - Установка
echo   Мультиязычный TTS (30 языков вкл. русский)
echo ========================================
echo.

REM Определяем директорию скрипта
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Локальные temp директории
set "TEMP=%SCRIPT_DIR%temp"
set "TMP=%SCRIPT_DIR%temp"

REM Создаем необходимые директории
if not exist "python" mkdir python
if not exist "downloads" mkdir downloads
if not exist "temp" mkdir temp
if not exist "models" mkdir models
if not exist "cache" mkdir cache
if not exist "output" mkdir output

echo [1/6] Выбор версии CUDA для вашей видеокарты...
echo.
echo Выберите поколение вашей видеокарты Nvidia:
echo.
echo 1. GTX 10xx серия (Pascal) - CUDA 11.8
echo 2. RTX 20xx серия (Turing) - CUDA 11.8
echo 3. RTX 30xx серия (Ampere) - CUDA 11.8 (стабильно, без Flash Attention 2)
echo 4. RTX 40xx серия (Ada Lovelace) - CUDA 12.8 + Flash Attention 2
echo 5. RTX 50xx серия (Blackwell) - CUDA 12.8 + Flash Attention 2
echo 6. CPU only (без GPU, медленнее)
echo.
set /p GPU_CHOICE="Введите номер (1-6): "

if "%GPU_CHOICE%"=="1" (
    set "CUDA_VERSION=cu118"
    set "CUDA_NAME=CUDA 11.8"
    set "TORCH_VERSION=2.7.1"
    set "TORCHAUDIO_VERSION=2.7.1"
)
if "%GPU_CHOICE%"=="2" (
    set "CUDA_VERSION=cu118"
    set "CUDA_NAME=CUDA 11.8"
    set "TORCH_VERSION=2.7.1"
    set "TORCHAUDIO_VERSION=2.7.1"
)
if "%GPU_CHOICE%"=="3" (
    set "CUDA_VERSION=cu118"
    set "CUDA_NAME=CUDA 11.8 (стабильно для Ampere)"
    set "TORCH_VERSION=2.7.1"
    set "TORCHAUDIO_VERSION=2.7.1"
)
if "%GPU_CHOICE%"=="4" (
    set "CUDA_VERSION=cu128"
    set "CUDA_NAME=CUDA 12.8"
    set "TORCH_VERSION=2.7.1"
    set "TORCHAUDIO_VERSION=2.7.1"
)
if "%GPU_CHOICE%"=="5" (
    set "CUDA_VERSION=cu128"
    set "CUDA_NAME=CUDA 12.8 (совместимо с RTX 50xx)"
    set "TORCH_VERSION=2.7.1"
    set "TORCHAUDIO_VERSION=2.7.1"
)
if "%GPU_CHOICE%"=="6" (
    set "CUDA_VERSION=cpu"
    set "CUDA_NAME=CPU only"
    set "TORCH_VERSION=2.8.0"
    set "TORCHAUDIO_VERSION=2.8.0"
)

if not defined CUDA_VERSION (
    echo Неверный выбор! Установка прервана.
    pause
    exit /b 1
)

echo.
echo Выбрано: %CUDA_NAME%
echo PyTorch: %TORCH_VERSION%
echo TorchAudio: %TORCHAUDIO_VERSION%
echo.

REM Проверяем наличие Python
if exist "python\python.exe" (
    echo [2/6] Python уже установлен, пропускаем загрузку...
) else (
    echo [2/6] Загрузка Python 3.12.8 Embeddable...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.8/python-3.12.8-embed-amd64.zip' -OutFile 'downloads\python-3.12.8-embed-amd64.zip'}"

    if not exist "downloads\python-3.12.8-embed-amd64.zip" (
        echo Ошибка загрузки Python!
        pause
        exit /b 1
    )

    echo Распаковка Python...
    powershell -Command "& {Expand-Archive -Path 'downloads\python-3.12.8-embed-amd64.zip' -DestinationPath 'python' -Force}"
)

REM Настраиваем Python для использования pip
echo [3/6] Настройка Python...
cd python

REM Удаляем ограничение импорта из python312._pth
if exist "python312._pth" (
    echo import site> python312._pth.new
    echo.>> python312._pth.new
    echo python312.zip>> python312._pth.new
    echo .>> python312._pth.new
    echo ..\Lib\site-packages>> python312._pth.new
    move /y python312._pth.new python312._pth >nul
)

cd ..

REM Устанавливаем pip
if exist "python\Scripts\pip.exe" (
    echo pip уже установлен
) else (
    echo Установка pip...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'downloads\get-pip.py'}"
    python\python.exe downloads\get-pip.py --no-warn-script-location
)

REM Обновляем pip
echo Обновление pip...
python\python.exe -m pip install --upgrade pip --no-warn-script-location

echo [4/6] Установка PyTorch %TORCH_VERSION% с %CUDA_NAME%...
python\python.exe -m pip install torch==%TORCH_VERSION% torchaudio==%TORCHAUDIO_VERSION% --index-url https://download.pytorch.org/whl/%CUDA_VERSION% --no-warn-script-location

echo [5/6] Установка зависимостей VoxCPM2...
python\python.exe -m pip install -r requirements.txt --no-warn-script-location

REM === Triton для Windows (torch.compile + кастомные kernel'ы) ===
REM Поддержка: все NVIDIA GPU (Pascal+), пропускаем на CPU
if not "%CUDA_VERSION%"=="cpu" (
    echo Установка Triton для Windows...
    python\python.exe -m pip install "triton-windows>=3.0.0,<3.4" --no-warn-script-location

    REM Python headers нужны Triton'у для компиляции launcher'а (embedded Python их не содержит)
    if not exist "python\Include\Python.h" (
        echo Установка Python headers для Triton...
        for /f "tokens=*" %%v in ('python\python.exe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"') do set "PY_VER=%%v"
        powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/!PY_VER!/amd64/dev.msi' -OutFile 'downloads\pydev.msi'}"
        if exist "downloads\pydev.msi" (
            msiexec /a "downloads\pydev.msi" /qn TARGETDIR="%SCRIPT_DIR%downloads\pydev_extract"
            if not exist "python\Include" mkdir "python\Include"
            if not exist "python\libs" mkdir "python\libs"
            xcopy /E /Y "downloads\pydev_extract\include\*" "python\Include\" >nul 2>&1
            xcopy /E /Y "downloads\pydev_extract\libs\*" "python\libs\" >nul 2>&1
            if exist "downloads\pydev_extract" rmdir /s /q "downloads\pydev_extract"
            echo Python headers установлены
        )
    )
)

REM === xformers (memory_efficient_attention) ===
REM Пинимся на 0.0.31.post1 — единственная версия совместимая с torch 2.7.1.
REM Без пина pip ставит latest (0.0.35), которая тянет torch 2.11 и сносит весь стек.
REM Сначала сносим возможный мусор от предыдущих запусков, потом ставим нужную.
REM Поддержка: все NVIDIA GPU, пропускаем на CPU
if not "%CUDA_VERSION%"=="cpu" (
    echo Чистка возможной старой xformers / torch из прошлых запусков...
    python\python.exe -m pip uninstall xformers -y 2>nul
    REM Если прошлый запуск зацепил torch 2.11 — откатим его на нужную версию
    python\python.exe -m pip install torch==%TORCH_VERSION% torchaudio==%TORCHAUDIO_VERSION% --index-url https://download.pytorch.org/whl/%CUDA_VERSION% --no-warn-script-location --force-reinstall --no-deps
    echo Установка xformers 0.0.31.post1 ^(под torch 2.7.1^)...
    python\python.exe -m pip install "xformers==0.0.31.post1" --no-warn-script-location
    if errorlevel 1 (
        echo Не удалось установить xformers. Приложение будет работать без него.
    ) else (
        echo xformers установлен успешно
    )
)

echo [5.5/6] Установка Flash Attention 2 (опционально)...
echo.
echo Flash Attention 2 ускоряет генерацию в 2-3 раза!
echo Рекомендуется для RTX 30xx/40xx/50xx серий.
echo.

REM Устанавливаем flash-attn в зависимости от GPU
if "%GPU_CHOICE%"=="3" (
    echo Flash Attention 2 пропущен для RTX 30xx на CUDA 11.8
    echo   ^(нет совместимой wheel под cu118+torch2.7, приложение работает через SDPA^)
    echo   Ускорение вернётся автоматически когда появится подходящая сборка.
)
if "%GPU_CHOICE%"=="4" (
    echo Установка Flash Attention 2 для RTX 40xx ^(Ada Lovelace^)...
    python\python.exe -m pip install https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl --no-warn-script-location
    if errorlevel 1 (
        echo Не удалось установить Flash Attention 2. Приложение будет работать через SDPA.
    ) else (
        echo Flash Attention 2 установлен успешно!
    )
)
if "%GPU_CHOICE%"=="5" (
    echo Установка Flash Attention 2 для RTX 50xx ^(Blackwell SM 12.0^)...
    python\python.exe -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.11/flash_attn-2.8.3+cu128torch2.7-cp312-cp312-win_amd64.whl --no-warn-script-location
    if errorlevel 1 (
        echo Не удалось установить Flash Attention 2. Приложение будет работать через SDPA.
    ) else (
        echo Flash Attention 2 установлен успешно!
    )
)
if "%GPU_CHOICE%"=="1" (
    echo Flash Attention 2 не рекомендуется для GTX 10xx
)
if "%GPU_CHOICE%"=="2" (
    echo Flash Attention 2 не рекомендуется для RTX 20xx
)
if "%GPU_CHOICE%"=="6" (
    echo Flash Attention 2 пропущен - CPU mode
)

echo.

echo [5.9/6] Загрузка голосового пакета по умолчанию...
if not exist "voices" mkdir voices
if exist "voices\*.mp3" (
    echo Голоса уже установлены, пропускаем...
) else (
    echo Загрузка voice-pack.zip из HuggingFace...
    curl -L -o downloads\voice-pack.zip https://huggingface.co/datasets/nerualdreming/VibeVoice/resolve/main/voice-pack.zip
    if exist "downloads\voice-pack.zip" (
        echo Распаковка голосового пакета...
        powershell -Command "Expand-Archive -Path 'downloads\voice-pack.zip' -DestinationPath 'downloads' -Force"
        if exist "downloads\voice-pack" (
            xcopy /E /Y /Q downloads\voice-pack\* voices\
            rmdir /S /Q downloads\voice-pack
            echo Голосовой пакет установлен!
        )
    ) else (
        echo ПРЕДУПРЕЖДЕНИЕ: не удалось скачать голосовой пакет.
        echo Можно скачать позже из UI - кнопка "Скачать все 743 голоса".
    )
)

echo [6/7] Установка портативного FFmpeg...
if exist "ffmpeg\ffmpeg.exe" (
    echo FFmpeg уже установлен, пропускаем...
) else (
    echo Загрузка FFmpeg...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip' -OutFile 'downloads\ffmpeg.zip'}"

    if not exist "downloads\ffmpeg.zip" (
        echo Ошибка загрузки FFmpeg! Попробуем альтернативный источник...
        powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile 'downloads\ffmpeg.zip'}"
    )

    if exist "downloads\ffmpeg.zip" (
        echo Распаковка FFmpeg...
        powershell -Command "& {Expand-Archive -Path 'downloads\ffmpeg.zip' -DestinationPath 'downloads\ffmpeg_temp' -Force}"

        REM Находим папку с ffmpeg.exe и копируем bin
        if not exist "ffmpeg" mkdir ffmpeg
        for /d %%D in (downloads\ffmpeg_temp\*) do (
            if exist "%%D\bin\ffmpeg.exe" (
                copy "%%D\bin\ffmpeg.exe" "ffmpeg\" >nul
                copy "%%D\bin\ffprobe.exe" "ffmpeg\" >nul
                echo FFmpeg установлен успешно!
            )
        )

        REM Очистка временных файлов
        rmdir /s /q "downloads\ffmpeg_temp" 2>nul
    ) else (
        echo ВНИМАНИЕ: Не удалось загрузить FFmpeg.
        echo Вы можете скачать его вручную с https://ffmpeg.org/download.html
        echo и распаковать ffmpeg.exe и ffprobe.exe в папку ffmpeg\
    )
)

echo [7/7] Финализация установки...
REM Создаем конфигурационный файл с версией CUDA
echo %CUDA_VERSION%> cuda_version.txt

echo.
echo ========================================
echo   Установка завершена успешно!
echo ========================================
echo.
echo Структура папок:
echo   models\  - кэш моделей HuggingFace (VoxCPM2 ~4-5 GB при первом запуске)
echo   output\  - сгенерированные WAV-файлы
echo   temp\    - временные файлы
echo   cache\   - кэш приложения
echo.
echo Для запуска приложения используйте: run.bat
echo.
pause
